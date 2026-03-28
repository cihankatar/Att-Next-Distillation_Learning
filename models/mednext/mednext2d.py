from models.mednext.MedNeXt_Block import MedNeXtBottleneckBlock,MedNeXtStem, MedNeXtDown, MedNeXtBlock, MedNeXtDeepSupervision, MedNeXtUp
import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis, parameter_count_table


class MedNeXtEncoder(nn.Module):
    def __init__(self, in_channels=3, C=64, encoder_blocks=[1,1,1,1], encoder_expansion=[4,4,4,4]):
        super(MedNeXtEncoder, self).__init__()
        self.C = C
        assert len(encoder_blocks) == 4, "encoder_blocks should have 4 elements."
        assert len(encoder_expansion) == 4, "encoder_expansion should have 4 elements."
        
        # Stem
        self.stem = MedNeXtStem(in_channels=in_channels, out_channels=C)
        
        # Encoder layers
        self.encoder1 = self._make_layer(MedNeXtBlock, C, encoder_blocks[0], encoder_expansion[0])
        self.encoder2 = nn.Sequential(
            MedNeXtDown(in_channels=C, expansion=encoder_expansion[0]),
            self._make_layer(MedNeXtBlock, 2*C, encoder_blocks[1], encoder_expansion[1])
        )
        self.encoder3 = nn.Sequential(
            MedNeXtDown(in_channels=2*C, expansion=encoder_expansion[1]),
            self._make_layer(MedNeXtBlock, 4*C, encoder_blocks[2], encoder_expansion[2])
        )
        self.encoder4 = nn.Sequential(
            MedNeXtDown(in_channels=4*C, expansion=encoder_expansion[2]),
            self._make_layer(MedNeXtBlock, 8*C, encoder_blocks[3], encoder_expansion[3])
        )

    def _make_layer(self, block, out_channels, blocks, expansion):
        layers = []
        for _ in range(blocks):
            layers.append(block(out_channels, expansion=expansion))
        return nn.Sequential(*layers)

    def forward(self, x):
        skip_connections = []

        # Stem
        x = self.stem(x)  # (B, C, H, W)
        
        # Encoder layers
        x = self.encoder1(x)  # (B, C, H, W)
        skip_connections.append(x)
        
        x = self.encoder2(x)  # (B, 2C, H/2, W/2)
        skip_connections.append(x)
        
        x = self.encoder3(x)  # (B, 4C, H/4, W/4)
        skip_connections.append(x)
        
        x = self.encoder4(x)  # (B, 8C, H/8, W/8)
        skip_connections.append(x)


        return x, skip_connections[::-1]  # Reverse the skip connections for the decoder


class MedNeXtDecoder(nn.Module):
    def __init__(self, in_channels, skip_channels, decoder_blocks, decoder_expansion, num_classes):
        super(MedNeXtDecoder, self).__init__()

        # Decoder stages (including the final segmentation head stage)
        self.decoder_stages = nn.ModuleList([
            nn.Sequential(
                self._make_layer(MedNeXtBlock, in_channels, decoder_blocks[0], decoder_expansion[0]),
                MedNeXtUp(in_channels=in_channels, expansion=decoder_expansion[0])
            ),
            nn.Sequential(
                self._make_layer(MedNeXtBlock, skip_channels[0], decoder_blocks[1], decoder_expansion[1]),
                MedNeXtUp(in_channels=skip_channels[0], expansion=decoder_expansion[1])
            ),
            nn.Sequential(
                self._make_layer(MedNeXtBlock, skip_channels[1], decoder_blocks[2], decoder_expansion[2]),
                MedNeXtUp(in_channels=skip_channels[1], expansion=decoder_expansion[2])
            ),
            nn.Sequential(
                self._make_layer(MedNeXtBlock, skip_channels[2], decoder_blocks[3], decoder_expansion[3]),
                MedNeXtUp(in_channels=skip_channels[2], expansion=decoder_expansion[3])
            ),
            # Final segmentation head stage
            nn.Sequential(
                self._make_layer(MedNeXtBlock, skip_channels[3], decoder_blocks[4], decoder_expansion[4]),
                MedNeXtDeepSupervision(in_channels=skip_channels[3], num_classes=num_classes)
            )
        ])
        
        # 1x1 convolution to reduce channels after concatenation
        self.conv1x1_layers = nn.ModuleList([
            nn.Conv2d(skip_channels[0] * 2, skip_channels[0], kernel_size=1, bias=False),
            nn.Conv2d(skip_channels[1] * 2, skip_channels[1], kernel_size=1, bias=False),
            nn.Conv2d(skip_channels[2] * 2, skip_channels[2], kernel_size=1, bias=False),
            nn.Conv2d(skip_channels[3] * 2, skip_channels[3], kernel_size=1, bias=False)
        ])
        
        # Deep Supervision layers for intermediate outputs
        self.deep_supervision_layers = nn.ModuleList([
            MedNeXtDeepSupervision(in_channels=skip_channels[0], num_classes=num_classes),
            MedNeXtDeepSupervision(in_channels=skip_channels[1], num_classes=num_classes),
            MedNeXtDeepSupervision(in_channels=skip_channels[2], num_classes=num_classes),
            MedNeXtDeepSupervision(in_channels=skip_channels[3], num_classes=num_classes)
        ])

    def _make_layer(self, block, out_channels, blocks, expansion):
        layers = []
        for _ in range(blocks):
            layers.append(block(out_channels, expansion=expansion))
        return nn.Sequential(*layers)

    def forward(self, x, skip_connections):
        deep_supervision_outputs = []
        
        # Decoder stages
        for i in range(4):
            skip = skip_connections[i]
            # Pass through MedNeXt block and MedNeXtUp
            x = self.decoder_stages[i](x)
            
            # Concatenate with skip connection
            x = torch.cat([x, skip], dim=1)
            
            # 1x1 convolution to reduce channels from 2 * skip_channels[i] to skip_channels[i]
            x = self.conv1x1_layers[i](x)
            
            # Apply Deep Supervision
            deep_supervision_output = self.deep_supervision_layers[i](x)
            deep_supervision_outputs.append(deep_supervision_output)
        
        # Final segmentation head stage
        final_output = self.decoder_stages[4](x)
        
        return final_output, deep_supervision_outputs


class MedNeXtSegmentationModel(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super(MedNeXtSegmentationModel, self).__init__()
        
        self.encoder = MedNeXtEncoder(
            in_channels=in_channels,
            C=32,
            encoder_blocks=[1, 1, 1, 1],  # 3,4,8,8
            encoder_expansion=[3, 4, 8, 8]
        )
        
        self.decoder = MedNeXtDecoder(
            in_channels=512,
            skip_channels=[256, 128, 64, 32],
            decoder_blocks=[1, 1, 1, 1,1],  # 8,8,8,4,3
            decoder_expansion=[8, 8, 8, 4, 3],
            num_classes=num_classes
        )
        self.bootleneck = MedNeXtBottleneckBlock( in_channels=256, kernel_size=5, expansion=4, groups=32, downsample=True)

    def forward(self, x):
        features, skip_connections = self.encoder(x)
        features = self.bootleneck(features)
        output = self.decoder(features, skip_connections)
        return output

if __name__ == '__main__':
    model = MedNeXtSegmentationModel()

    # 创建测试输入张量
    input_tensor = torch.randn(8, 3, 224, 224)
    output = model(input_tensor)
    print(output[0].shape)
