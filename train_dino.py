import os
import torch
import wandb
import copy
from tqdm import tqdm, trange
from torch.optim import AdamW 
from torch.optim.lr_scheduler import CosineAnnealingLR
from data.data_loader import loader
import torch.nn.functional as F
from wandb_init import parser_init, wandb_init
from models.Model import model_dice_bce

from models.Unet import UNET
from utils.Loss_dino import DINOLoss
from torch.nn.utils import clip_grad_norm_
from torch import nn 
import matplotlib.pyplot as plt
import numpy as np
import time

def get_teacher_momentum(current_epoch, max_epochs, base_m=0.996, final_m=1.0):
    # Linear momentum schedule
    return base_m + (final_m - base_m) * (current_epoch / max_epochs)

def get_teacher_temp(epoch, warmup_epochs=20, final_temp=0.07):
    start_temp = 0.04
    if epoch < warmup_epochs:
        return start_temp + (final_temp - start_temp) * epoch / warmup_epochs
    else:
        return final_temp
    
class ProjectionHead(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=1024, out_dim=4096):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),)
    def forward(self, x):
        return self.mlp(x)

class SegmentationSHead(nn.Module):
    def __init__(self, in_channels=512, out_channels=1, final_size=(256, 256)):
        super().__init__()
        self.final_size = final_size
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=1)
        )
    def forward(self, x):
        x = self.head(x)  # Shape: (B, 1, 8, 8)
        x = F.interpolate(x, size=self.final_size, mode='bilinear', align_corners=False)
        return x

class SegmentationTHead(nn.Module):
    def __init__(self, in_channels=512, out_channels=1, final_size=(256, 256)):
        super().__init__()
        self.final_size = final_size
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=1)
        )
    def forward(self, x):
        x = self.head(x)  # Shape: (B, 1, 8, 8)
        x = F.interpolate(x, size=self.final_size, mode='bilinear', align_corners=False)
        return x

def featuremap_to_heatmap(tensor):
    """Convert a 2D tensor to a colored heatmap suitable for wandb.Image."""
    heatmap = tensor.detach().cpu()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-5)  # Normalize to [0, 1]
    heatmap_np = heatmap.numpy()

    # Apply colormap
    cmap = plt.get_cmap("viridis")  # Choose any matplotlib colormap: 'jet', 'plasma', etc.
    colored_heatmap = cmap(heatmap_np)[:, :, :3]  # Drop alpha channel

    # Convert to 8-bit RGB
    colored_heatmap = (colored_heatmap * 255).astype(np.uint8)
    return colored_heatmap

def using_device():
    """Set and print the device used for training."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
    return device

def setup_paths(data):
    """Set up data paths for training and validation."""
    folder_mapping = {
        "isic_2018_1": "isic_1/",
        "kvasir_1": "kvasir_1/",
        "ham_1": "ham_1/",
        "PH2Dataset": "PH2Dataset/",
        "isic_2016_1": "isic_2016_1/"
    }
    folder = folder_mapping.get(data)
    base_path = os.environ["ML_DATA_OUTPUT"] if torch.cuda.is_available() else os.environ["ML_DATA_OUTPUT_LOCAL"]
    print(base_path)
    return os.path.join(base_path, folder)

@torch.no_grad()
def update_teacher(student, teacher, momentum):
    for param_s, param_t in zip(student.parameters(), teacher.parameters()):
        param_t.data = momentum * param_t.data + (1. - momentum) * param_s.data

def main():

    data, training_mode, op = 'isic_2018_1', "ssl", "train"

    best_loss   = float("inf")
    device      = using_device()
    folder_path = setup_paths(data)
    args, res   = parser_init("segmentation task", op, training_mode)
    res         = " ".join(res)
    res         = "["+res+"]"

    config      = wandb_init(os.environ["WANDB_API_KEY"], os.environ["WANDB_DIR"], args, data)
    args.suffle         = False
    print("train_im_path", os.environ["ML_DATA_ROOT"]+"train/images") 

    # Data Loaders
    def create_loader(operation):

        return loader(operation,args.mode, args.sslmode_modelname, args.bsize, args.workers,
                      args.imsize, args.cutoutpr, args.cutoutbox, args.shuffle, args.sratio, data)

    train_loader    = create_loader(args.op)
    args.op         =  "validation"
    
    val_loader      = create_loader(args.op)
    args.op         = "train"
    # Student & Teacher modeli
    #model     = UNET(1).to(device)
    model     = model_dice_bce().to(device)

    s_head    = SegmentationSHead().to(device)
    t_head    = SegmentationTHead().to(device)

    student = model.encoder
    teacher = copy.deepcopy(student)
    teacher = teacher.to(device)
    teacher.load_state_dict(student.state_dict())

    for p in teacher.parameters():
        p.requires_grad = False

    # Optimizasyon & Loss

    loss_fn         = DINOLoss()
    checkpoint_path = folder_path+str(student.__class__.__name__)+str(res)
    optimizer       = AdamW(student.parameters(), lr=config['learningrate'],weight_decay=0.05)
    scheduler       = CosineAnnealingLR(optimizer, config['epochs'], eta_min=config['learningrate'] / 10)
    
    print(f"Training on {len(train_loader) * args.bsize} images. Saving checkpoints to {folder_path}")
    print(f"model config : {checkpoint_path}")

    ML_DATA_OUTPUT      = os.environ["ML_DATA_OUTPUT"]+'isic_1/'

    checkpoint_path_read = ML_DATA_OUTPUT+str(student.__class__.__name__)+str(res)
    #student.load_state_dict(torch.load(checkpoint_path_read, map_location=torch.device('cpu')))

    checkpoint_path_head = ML_DATA_OUTPUT+str(s_head.__class__.__name__)+str(res)
    #s_head.load_state_dict(torch.load(checkpoint_path_head, map_location=torch.device('cpu')))

    student_head = ProjectionHead().to(device)
    teacher_head = copy.deepcopy(student_head).to(device)
    
    for p in teacher_head.parameters():
        p.requires_grad = False
    # Training and Validation Loops

    def run_epoch(loader, epoch_idx, momentum, training=True):
        epoch_loss  = 0.0
        num_batches = 0
        epoch_val_loss, epoch_seg_loss = 0.0, 0.0
        weigt       = 0.1
        student.train() if training else student.eval()
        teacher.eval()

        teacher_temp = get_teacher_temp(epoch_idx)
        current_lr = optimizer.param_groups[0]['lr']
        print("teacher temp", teacher_temp, "\n")
        print("current_lr", current_lr, "\n")
        with torch.set_grad_enabled(training):
            for img,student_augs, teacher_augs, pseudo_masks in loader:
                end_time = time.time()

                student_feats = [student(im.to(device))[3] for im in student_augs]
                student_pool  = [feat.mean(dim=(2, 3)) for feat in student_feats]
                student_proj  = [F.normalize(student_head(p), dim=1) for p in student_pool]

                with torch.no_grad():

                    teacher_feats = [teacher(im.to(device))[3] for im in teacher_augs]
                    teacher_pool  = [feat.mean(dim=(2, 3)) for feat in teacher_feats]
                    teacher_proj  = [F.normalize(teacher_head(p), dim=1) for p in teacher_pool]

                seg_feats  = student_feats[0]
                seg_logits = s_head(seg_feats)              # Shape: [B, 512, 8, 8]
                seg_target = pseudo_masks[0].to(device).type_as(seg_logits)              # Shape: [B, 1, 8, 8]

                # Segmentation head prediction
                # student_views = [s_head(feat)  for feat in student_feats]           # Shape: [B, 512, 8, 8]
                # teacher_views =  [t_head(feat)  for feat in teacher_feats]           # Shape: [B, 512, 16, 16]


                # Optionally: interpolate seg_target if output is different spatially (here it's same 8x8)
                # seg_target = F.interpolate(seg_target, size=seg_logits.shape[-2:], mode='bilinear', align_corners=False)

                # Compute segmentation loss

                #seg_loss = F.binary_cross_entropy_with_logits(seg_logits, seg_target)
                loss = loss_fn(student_proj, teacher_proj, teacher_temp)
                #loss = loss_c * weigt + seg_loss


                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                update_teacher(student, teacher, momentum)
                update_teacher(student_head, teacher_head, momentum)
                update_teacher(s_head, t_head, momentum)
                
                epoch_loss += loss.item()
                epoch_seg_loss += seg_loss.item()

                if not training:
                    # Cosine similarity between projected features
                    stu_stack = torch.stack(student_proj, dim=1)  # [B, views_s, C]
                    tea_stack = torch.stack(teacher_proj, dim=1)  # [B, views_t, C]

                    pairwise = []
                    for s in range(stu_stack.size(1)):
                        for t in range(tea_stack.size(1)):
                            cos = F.cosine_similarity(stu_stack[:, s], tea_stack[:, t], dim=-1)
                            pairwise.append(cos)

                    val_loss = torch.stack(pairwise, dim=1).mean()
                    epoch_val_loss += val_loss.item()

                    # Log heatmaps and augs for first batch only
                    if num_batches == 0:
                        b_idx, v_idx = 0, 0
                        student_map = student_feats[v_idx][b_idx].mean(dim=0).detach().cpu()
                        teacher_map = teacher_feats[v_idx][b_idx].mean(dim=0).detach().cpu()

                        pseudo_mask  = seg_target[b_idx].permute(1,2,0).detach().cpu().numpy()
                        seg_logit    = seg_logits[b_idx].permute(1,2,0).detach().cpu().numpy() 
                        student_heatmap = featuremap_to_heatmap(student_map)
                        teacher_heatmap = featuremap_to_heatmap(teacher_map)
                        student_aug  = student_augs[v_idx][b_idx].permute(1,2,0).detach().cpu().numpy()
                        teacher_aug  = teacher_augs[v_idx][b_idx].permute(1,2,0).detach().cpu().numpy()   
                        im           = img[b_idx].permute(1,2,0).detach().cpu() 

                        wandb.log({
                            "Val Sample - Student Aug": wandb.Image(student_aug),
                            "Val Sample - Teacher Aug": wandb.Image(teacher_aug),
                            "Val Sample - Student Output Heatmap": wandb.Image(student_heatmap),
                            "Val Sample - Teacher Output Heatmap": wandb.Image(teacher_heatmap),
                            "Val Sample - Pseudo Segmentation Mask": wandb.Image(pseudo_mask),
                            "Val Sample - Pseudo Prediction Mask": wandb.Image(seg_logit),
                        })

                num_batches += 1

        if not training:
            return epoch_val_loss / len(loader)

        return epoch_loss / len(loader), epoch_seg_loss / len(loader)

    epoch_idx=0
    for epoch in trange(config['epochs'], desc="Epochs"):

        # Training
        current_momentum = get_teacher_momentum(epoch, config['epochs'])
        train_loss,seg_loss = run_epoch(train_loader, epoch_idx, current_momentum,training=True )
        wandb.log({"Train Loss": train_loss, "seg_loss": seg_loss})
        scheduler.step()

        cos_sim = run_epoch(val_loader, epoch_idx,current_momentum,training=False)
        wandb.log({"Cosine Similarity": cos_sim })

        epoch_idx+=1

        print("epoch_idx",epoch_idx,"\n")
        
        # Print losses and validation metrics
        print(f"Train Loss: {train_loss:.4f}, Segmentation Loss : {seg_loss:.4f}")
        print(f"Validation Cosine Similarity: {cos_sim:.4f}")

        # Save best model
        if epoch_idx>50:
            if train_loss < best_loss:
                best_loss = train_loss
                torch.save(student.state_dict(), checkpoint_path)
                print(f"Best model saved")

    wandb.finish()

if __name__ == "__main__":
    main()
