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
from utils.Heads import ProjectionHead, SegmentationSHead, SegmentationMHead, get_teacher_momentum, get_teacher_temp
from models.Unet import UNET
from utils.Loss_dino import DINOLoss
from torch.nn.utils import clip_grad_norm_
from torch import nn 
import matplotlib.pyplot as plt
import numpy as np
import time


def compute_batch_iou(pred_logits, target_masks):
    """
    Batch bazında ortalama IoU hesaplar.
    pred_logits: [B, 1, H, W]
    target_masks: [B, 1, H, W] (0 veya 1 değerleri)
    """
    # Sigmoid ve Threshold ile Binary Maske
    probs = torch.sigmoid(pred_logits)
    preds = (probs > 0.5).float()
    
    # Target maskeleri binary olduğundan emin olalım
    targets = (target_masks > 0.5).float()
    
    # Kesişim ve Birleşim (Batch boyunca topla değil, her örnek için ayrı hesapla sonra ortala)
    # Düzleştirelim: [B, -1]
    preds_flat = preds.view(preds.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)
    
    intersection = (preds_flat * targets_flat).sum(dim=1)  # [B]
    union = (preds_flat + targets_flat).sum(dim=1) - intersection # [B]
    
    # Sıfıra bölünmeyi engelle (smooth)
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    return iou.mean().item() # Batch ortalaması

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

    data, training_mode, op, dinowithsegloss = 'isic_2018_1', "ssl", "train",True

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
    monitor_head    = SegmentationMHead().to(device)

    student = model.encoder
    teacher = copy.deepcopy(student)
    teacher = teacher.to(device)
    teacher.load_state_dict(student.state_dict())

    for p in teacher.parameters():
        p.requires_grad = False

    # Optimizasyon & Loss

    loss_fn         = DINOLoss()
    checkpoint_path = folder_path+str(student.__class__.__name__)+str(res)

    params_to_optimize = list(student.parameters())
    if dinowithsegloss:
        params_to_optimize += list(s_head.parameters())
    
    optimizer = AdamW(params_to_optimize, lr=config['learningrate'], weight_decay=0.05)
    scheduler       = CosineAnnealingLR(optimizer, config['epochs'], eta_min=config['learningrate'] / 10)

    ML_DATA_OUTPUT      = os.environ["ML_DATA_OUTPUT"]+'isic_1/'
    checkpoint_path_read = ML_DATA_OUTPUT+str(student.__class__.__name__)+str(res)
    #student.load_state_dict(torch.load(checkpoint_path_read, map_location=torch.device('cpu')))
    checkpoint_path_head = ML_DATA_OUTPUT+str(s_head.__class__.__name__)+str(res)
    #s_head.load_state_dict(torch.load(checkpoint_path_head, map_location=torch.device('cpu')))

    student_head = ProjectionHead().to(device)
    teacher_head = copy.deepcopy(student_head).to(device)
    # Segmentation Head (Online Probe) için ayrı optimizer
    # Bu kafa sadece detach edilmiş özelliklerle eğitilecek
    for p in teacher_head.parameters():
        p.requires_grad = False

    print(f"Training on {len(train_loader) * args.bsize} images. Saving checkpoints to {folder_path}")
    print(f"model config : {checkpoint_path}")
    #ONLY for monitor head
    optimizer_monitor  = AdamW(monitor_head.parameters(), lr=config['learningrate'])

    # Training and Validation Loops

    def run_epoch(loader, epoch_idx, momentum, training=True):
        epoch_loss  = 0.0
        num_batches = 0
        epoch_val_loss, epoch_seg_loss,epoch_monitor_loss, epoch_iou = 0.0, 0.0, 0.0, 0.0
        weigt       = 0.1

        if training:
            student.train()
            s_head.train()
            monitor_head.train()
        else:
            student.eval()
            s_head.eval()
            monitor_head.eval()
        teacher.eval()

        teacher_temp = get_teacher_temp(epoch_idx)
        current_lr = optimizer.param_groups[0]['lr']
        
        print("teacher temp", teacher_temp, "\n")
        print("current_lr", current_lr, "\n")

        with torch.set_grad_enabled(training):
            for img, path, cropped_real_mask, student_augs, teacher_augs, pseudo_masks in loader:

                student_feats = [student(im.to(device))[3] for im in student_augs]
                student_pool  = [feat.mean(dim=(2, 3)) for feat in student_feats]
                student_proj  = [F.normalize(student_head(p), dim=1) for p in student_pool]

                with torch.no_grad():

                    teacher_feats = [teacher(im.to(device))[3] for im in teacher_augs]
                    teacher_pool  = [feat.mean(dim=(2, 3)) for feat in teacher_feats]
                    teacher_proj  = [F.normalize(teacher_head(p), dim=1) for p in teacher_pool]

                seg_logits      = s_head(student_feats[0])              # Shape: [B, 512, 8, 8]
                seg_target      = pseudo_masks[0].to(device).type_as(seg_logits)              # Shape: [B, 1, 8, 8]
                real_seg_target = cropped_real_mask[0].to(device).type_as(seg_logits)              # Shape: [B, 1, 8, 8]

                # Compute segmentation loss
                if training:
                    if dinowithsegloss:
                        seg_loss  = F.binary_cross_entropy_with_logits(seg_logits, seg_target)
                        loss_c    = loss_fn(student_proj, teacher_proj, teacher_temp)
                        loss      = loss_c * weigt + seg_loss
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        epoch_seg_loss += seg_loss.item()

                    else:
                        loss = loss_fn(student_proj, teacher_proj, teacher_temp)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        epoch_seg_loss += 0.0


                    detached_features  = [f.detach() for f in student_feats]
                    seg_monitor        = monitor_head(detached_features[0])   
                    monitor_loss       = F.binary_cross_entropy_with_logits(seg_monitor, real_seg_target.type_as(seg_monitor))
                    optimizer_monitor.zero_grad()
                    monitor_loss.backward()
                    optimizer_monitor.step() 

                    epoch_loss += loss.item()
                    epoch_monitor_loss += monitor_loss.item()

                    update_teacher(student, teacher, momentum)
                    update_teacher(student_head, teacher_head, momentum)

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
                    
                    seg_monitor        = monitor_head(student_feats[0])   
                    iou = compute_batch_iou(seg_monitor, real_seg_target)
                    epoch_iou += iou

                    # Log heatmaps and augs for first batch only
                    if num_batches == 0:
                        b_idx, v_idx = 0, 0
                        p = path[b_idx]
                        student_feat = student_feats[v_idx][b_idx].mean(dim=0).detach().cpu()
                        teacher_feat = teacher_feats[v_idx][b_idx].mean(dim=0).detach().cpu()
                        pseudo_mask  = seg_target[b_idx].permute(1,2,0).detach().cpu().numpy()
                        if dinowithsegloss:
                            seg_logit    = seg_logits[b_idx].permute(1,2,0).detach().cpu().numpy()

                        student_heatmap = featuremap_to_heatmap(student_feat)
                        teacher_heatmap = featuremap_to_heatmap(teacher_feat)

                        student_aug  = student_augs[v_idx][b_idx].permute(1,2,0).detach().cpu().numpy()
                        teacher_aug  = teacher_augs[v_idx][b_idx].permute(1,2,0).detach().cpu().numpy() 

                        im                    = img[b_idx].permute(1,2,0).detach().cpu().numpy()
                        cropped_real_mask     = cropped_real_mask[v_idx][b_idx].permute(1,2,0).detach().cpu().numpy()

                        prob = torch.sigmoid(seg_monitor[b_idx])
                        pred_mask = (prob > 0.5).float().permute(1,2,0).detach().cpu().numpy() 
                        intersection = (pred_mask * cropped_real_mask).sum()
                        union = (pred_mask + cropped_real_mask - pred_mask * cropped_real_mask).sum()
                        iou = (intersection + 1e-6) / (union + 1e-6)

                        intersection = (pseudo_mask * cropped_real_mask).sum()
                        union = (pseudo_mask + cropped_real_mask - pseudo_mask * cropped_real_mask).sum()
                        iou_pseudo = (intersection + 1e-6) / (union + 1e-6)

                        wandb.log({
                            "Sample Image": wandb.Image(im, caption=f"{p}"),
                            "Cropped Real mask": wandb.Image(cropped_real_mask),
                            "Student Aug": wandb.Image(student_aug),
                            "Teacher Aug": wandb.Image(teacher_aug),
                            "Student Output Heatmap": wandb.Image(student_heatmap),
                            "Teacher Output Heatmap": wandb.Image(teacher_heatmap),
                            "Pseudo Segmentation Mask": wandb.Image(pseudo_mask, caption=f"IoU with Real Mask: {iou_pseudo:.4f}"),
                            "Monitor Head Prediction prob": wandb.Image(seg_monitor),
                            "Auxillary Seg Prediction prob": wandb.Image(seg_logit) if dinowithsegloss else None,
                            "Monitor Head Prediction Mask": wandb.Image(pred_mask, caption=f"IoU:{iou:.4f}"),
                        })

                num_batches += 1

        if not training:
            return epoch_val_loss / len(loader), epoch_iou / len(loader)

        return epoch_loss / len(loader), epoch_seg_loss / len(loader), epoch_monitor_loss / len(loader)

    epoch_idx=0
    for epoch in trange(config['epochs'], desc="Epochs"):

        # Training
        current_momentum = get_teacher_momentum(epoch, config['epochs'])
        train_loss,seg_loss,monitor_loss = run_epoch(train_loader, epoch_idx, current_momentum,training=True )
        wandb.log({"Train Loss": train_loss,
                    "seg_loss": seg_loss, 
                    "monitor_loss": monitor_loss})
        scheduler.step()

        cos_sim,val_iou = run_epoch(val_loader, epoch_idx,current_momentum,training=False)
        wandb.log({"Cosine Similarity": cos_sim, 
                   "Validation IoU": val_iou })

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
