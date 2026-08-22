import os
import argparse
import numpy as np
import cv2
from PIL import Image
import torch
from skimage import morphology as morph
import torchvision.transforms as T
from torch_topological.nn import CubicalComplex
from torchvision.transforms.functional import to_pil_image
from skimage.transform import resize
from skimage import color
from utils.Dullrazor import dullrazor
from utils.Algorithms import morphological_chan_vese_segmentation, random_walker_pseudo_mask
from utils.padding import adaptive_pad, unpad_resize
from skimage.filters import threshold_otsu
from plotting import plot_topology_results

def smart_h1_threshold(gray, birth, a, pers, steps=20):

    thresholds = np.linspace(birth, birth + 0.75 * pers, steps)
    areas = np.array([(gray >= th).sum() for th in thresholds])

    dy = np.gradient(areas)  
    ddy = np.gradient(dy)           
    der_idx = np.argmax(ddy)

    x = np.linspace(1, 0, len(areas))                   
    y = (areas - areas.min()) / (areas.max() - areas.min() + 1e-9)

    diff = np.abs(x - y)
    knee_idx = np.argmax(diff)         
    dx = x[-1] - x[0]
    dy = y[-1] - y[0]
    
    slope = dy / dx
    intercept = y[0] - slope * x[0]
    y_line = slope * x + intercept    
    distances = np.abs(y - y_line)
    
    knee_idx = np.argmax(distances)
    knee_strength = distances[knee_idx]
    if knee_strength < 0.20: 

        best_th = birth + 0.10 * pers
        return best_th

    best_th = thresholds[(der_idx+2)]

    return best_th

def cubical_complex_segmentation(image, prcntg, h0=True,persistence_threshold=20.0):

    if image.ndim==3:  
        img=image[:,:,0] 
        lifetime=image[:,:,1] 
        gray = 255 - img

    else:  
        img = image
        gray =  255-img

    # Build cubical complex and compute persistence using torch_topological
    torch_cubical_complex = CubicalComplex(superlevel=False)
    pi_1  = torch_cubical_complex(torch.from_numpy(img))
    pi_0  = torch_cubical_complex(torch.from_numpy(gray))

    # death - birth hesapla (inf olursa inf çıkacak)
    persistence_h1 = pi_1[1][1][:,1] - pi_1[1][1][:,0]
    persistence_h0 = pi_0[0][1][:,1] - pi_0[0][1][:,0]

    # (birth, death, persistence) olarak birleştir
    pi_h1_full = torch.cat([pi_1[1][1], persistence_h1.unsqueeze(1)], dim=1)
    pi_h0_full = torch.cat([pi_0[0][1], persistence_h0.unsqueeze(1)], dim=1)

    # sonsuz persistansları (ölmeyenler) filtrele
    finite_mask = torch.isfinite(pi_h1_full[:,1])
    pi_finite_h1 = pi_h1_full[finite_mask]

    # persistence’e göre büyükten küçüğe sırala h1
    pi_sorted_h1 = pi_finite_h1[torch.argsort(pi_finite_h1[:,2], descending=True)]
    best_birth, best_death, best_pers = pi_sorted_h1[0]

    pi_sorted_h0 = pi_h0_full[torch.argsort(pi_h0_full[:,2], descending=True)]
    best_birth_h0, best_death_ho, best_pers_ho = pi_sorted_h0[1]

    # --- Determine tau for H0 ---  
    tau_h0 = None
    if len(pi_sorted_h0) > 0:
        lifetimes_h0 = pi_sorted_h0[:, 2].detach().cpu().numpy()
        lifetimes_h0 = lifetimes_h0[1:]
        if len(lifetimes_h0) > 1:
            # ardışık persistans farkları
            diffs = lifetimes_h0[:-1] - lifetimes_h0[1:]
            max_gap_idx = np.argmax(diffs)
            tau_h0 = (lifetimes_h0[max_gap_idx] + lifetimes_h0[max_gap_idx + 1]) / 2
        else:
            tau_h0 = persistence_threshold if persistence_threshold else lifetimes_h0[0] / 2

        # --- build segmentation map ---
        seg_map = np.zeros_like(img, dtype=np.uint8)
        
        pi_sorted_h0 = pi_sorted_h0[1:]  # ilk elemanı at (0,255)
        for birth, death, pers in pi_sorted_h0:
            if pers.item() >= tau_h0:
                mask = ((gray) >= birth.item()) & ((gray) <= death.item())
                seg_map = np.logical_or(seg_map, mask)

    final_seg_map = morph.remove_small_holes(seg_map.astype(bool), area_threshold=32*32).astype(np.uint8)
    mask_h0 = morph.remove_small_objects(final_seg_map.astype(bool), min_size=170).astype(np.uint8)

    p = best_pers.numpy()
    p_norm = p / np.max([p, 255])  
    adapt_factor = (1 - np.log1p(p_norm)) * 10
    #adapt_factor = np.log1p(best_pers.numpy())  # log(1+pers)
    ## h1 hard threshoulding ###
    th  = best_birth.numpy()+best_pers.numpy()/adapt_factor   
    mask = (img >= th)
    mask_h1 = mask.astype(np.uint8)
    mask_h1 = morph.remove_small_holes(mask_h1.astype(bool), area_threshold=32*32).astype(np.uint8)
    final_mask = morph.remove_small_objects(mask_h1.astype(bool), min_size=64).astype(np.uint8)

    holl = (img >= th) & (img <= best_death.numpy() - best_pers.numpy()/prcntg) 
    final_mask_holl = morph.remove_small_holes(holl.astype(bool), area_threshold=100*100).astype(np.uint8)
    final_mask_holl = morph.remove_small_objects(final_mask_holl.astype(bool), min_size=64).astype(np.uint8)

    th_smart = smart_h1_threshold(img, best_birth.numpy(),best_pers.numpy()/adapt_factor, best_pers.numpy())
    mask_smart = (img >= th_smart)
    mask_smart_final = mask_smart.astype(np.uint8)
    mask_smart_final = morph.remove_small_holes(mask_smart_final.astype(bool), area_threshold=32*32).astype(np.uint8)
    mask_smart_final = morph.remove_small_objects(mask_smart_final.astype(bool), min_size=64).astype(np.uint8)

    return mask_smart_final, pi_1, pi_sorted_h0[0].numpy(),pi_sorted_h0[1].numpy(),pi_sorted_h1[0].numpy(),final_mask_holl,final_mask,mask_h0,th

def otsu_weight(channel):
    t = threshold_otsu(channel)
    fg = channel[channel > t]
    bg = channel[channel <= t]
    w1, w2 = len(fg)/len(channel.flatten()), len(bg)/len(channel.flatten())
    return w1 * w2 * (fg.mean() - bg.mean())**2

def generate_pseudo_mask(
    image_path,
    mask_path=None,
    output_path=None,
    index=0,
    size=(256, 256),
    show=False,
):

    pil_img = Image.open(image_path).convert("RGB").resize(size)
    real_mask = None
    if mask_path is not None:
        real_mask = Image.open(mask_path).convert("L").resize(size)
        real_mask = np.array(real_mask)

    # Convert to tensor (automatically scales 0–255 → 0–1 and makes shape [C,H,W])
    # Apply dullrazor (assuming it expects [C,H,W] tensor in float32, 0–1 range)

    to_tensor = T.ToTensor()
    img_tensor = to_tensor(pil_img)
    clean_img = dullrazor(img_tensor)
    
    clean_img = clean_img.permute(1, 2, 0).cpu().numpy()

    arr = np.array(to_pil_image(clean_img).convert("L"))

    hsv = cv2.cvtColor(clean_img, cv2.COLOR_RGB2HSV)
    lab = color.rgb2lab(clean_img)
    S, V = hsv[:,:,1], hsv[:,:,2]
    A, B = lab[:,:,1], lab[:,:,2]
    eps = 1e-8
    A_norm = (A - A.min()) / (A.max() - A.min() + eps)
    B_norm = (B - B.min()) / (B.max() - B.min() + eps)
    S_norm = (S - S.min()) / (S.max() - S.min() + eps)
    V_norm = 1 - (V - V.min()) / (V.max() - V.min() + eps)
    gray = (0.2 * A_norm + 0.2 * B_norm + 0.3 * S_norm+ 0.3 * V_norm )*255

    #pca_ch_ = pca_channel(img, gray)*255

    pad_size = 5
    padded   = adaptive_pad(gray, pad_size,5)
    rpadded  = resize(padded, (256, 256),order=1, preserve_range=True, anti_aliasing=True).astype(gray.dtype)

    input_image = rpadded
    images = [("pca_ch", rpadded)]

    morphological_mask, _ = morphological_chan_vese_segmentation(arr)
    random_walker_mask    = random_walker_pseudo_mask(arr)

    th = threshold_otsu(gray)
    otsu_mask = gray > th

    for id, (name,img_input) in enumerate(images, start=1):
        cc_mask, pi, _,_, bests, mask_holl,mask_h1,mask_ho, th = cubical_complex_segmentation(input_image, prcntg=5, persistence_threshold=1.0)
        cc_mask = unpad_resize(cc_mask, orig_shape=gray.shape, pad=pad_size, target_size=256)
        mask_h1 = unpad_resize(mask_h1, orig_shape=gray.shape, pad=pad_size, target_size=256)
        if output_path is not None:
            Image.fromarray((cc_mask > 0).astype(np.uint8) * 255).save(output_path)

        if show:
            plot_topology_results(
                name,
                img_input,
                random_walker_mask,
                morphological_mask,
                cc_mask,
                real_mask,
                img_tensor,
                clean_img,
                bests,
                otsu_mask,
                mask_holl,
                mask_h1,
                mask_ho,
                pi,
                th,
            )
        print("processed", index, image_path)
        return cc_mask


def parse_args():
    parser = argparse.ArgumentParser(description="Generate persistent-homology pseudo-masks")
    parser.add_argument(
        "--dataset",
        default=os.environ.get("TOPODISTILL_DATASET", "kvasir_1"),
        help="dataset directory name under ML_DATA_ROOT",
    )
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--show", action="store_true", help="show diagnostic plots")
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = os.environ.get("ML_DATA_ROOT")
    if not data_root:
        raise EnvironmentError("ML_DATA_ROOT must point to the directory containing the datasets")

    split_path = os.path.join(data_root, args.dataset, args.split)
    image_path = os.path.join(split_path, "images")
    mask_path = os.path.join(split_path, "masks")
    pseudo_mask_path = os.path.join(split_path, "pmasks")
    os.makedirs(pseudo_mask_path, exist_ok=True)

    images = sorted(
        name for name in os.listdir(image_path) if name.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    masks = []
    if os.path.isdir(mask_path):
        masks = sorted(
            name for name in os.listdir(mask_path) if name.lower().endswith((".jpg", ".jpeg", ".png"))
        )

    selected = images[args.start_index :]
    if args.limit is not None:
        selected = selected[: args.limit]

    for offset, image_name in enumerate(selected, start=args.start_index):
        current_image = os.path.join(image_path, image_name)
        current_mask = os.path.join(mask_path, masks[offset]) if offset < len(masks) else None
        output_name = f"{os.path.splitext(image_name)[0]}.png"
        output_file = os.path.join(pseudo_mask_path, output_name)
        print(f"[{offset + 1}/{len(images)}] Processing {image_name}")
        generate_pseudo_mask(
            current_image,
            current_mask,
            output_file,
            index=offset,
            show=args.show,
        )

if __name__ == "__main__":
    main()
