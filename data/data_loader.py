from torch.utils.data import DataLoader
from data.Custom_Dataset import dataset
from glob import glob
from torchvision.transforms import v2 
import os
import torch
import numpy as np
import skimage.segmentation as seg
import skimage.filters as filters
import skimage.morphology as morph
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

def pca_channel(img, gray):
    """
    Apply PCA to reduce multi-channel image (2 or 3 channels) to 1 channel.
    Ensure PCA direction aligns with reference gray image (same bright/dark orientation).
    """
    h, w, c = img.shape
    flat = img.reshape(-1, c).astype(np.float32)
    
    # PCA -> 1 bileşen
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(flat)
    pc1 = (pc1 - pc1.min()) / (pc1.max() - pc1.min() + 1e-12)
    pc1_img = pc1.reshape(h, w)
    
    # --- Gray ile yön hizalama ---
    # g = gray.astype(np.float32)
    # g = (g - g.min()) / (g.max() - g.min() + 1e-12)
    
    corr = np.corrcoef(pc1_img.ravel(), gray.ravel())[0, 1]
    if corr < 0:
        pc1_img = 1 - pc1_img  # yön ters ise düzelt
        
    return pc1_img

def random_walker_pseudo_mask(img_tensor):
    """
    Convert a PyTorch tensor [3, H, W] to a 2D pseudo‐mask via Random Walker.
    Steps:
      1) Convert to grayscale
      2) Compute Otsu threshold on grayscale -> get a rough binary
      3) Define “foreground” and “background” seeds for Random Walker
      4) Run Random Walker to refine the mask
      5) Post‐process with small morphological operations
    Returns:  torch.Tensor of shape [1, H, W], dtype=torch.float32, values 0.0 or 1.0
    """
    # (a) Convert to NumPy grayscale [H, W]
    C,H,W = img_tensor.shape
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()  
    gray = 0.2989 * img_np[..., 0] + 0.5870 * img_np[..., 1] + 0.1140 * img_np[..., 2]

    # (b) Rough Otsu threshold
    thresh = filters.threshold_otsu(gray)
    rough_binary = (gray < thresh).astype(np.uint8)  
    delta = 0.10 * (gray.max() - gray.min())
    markers = np.zeros_like(gray, dtype=np.int32)
    markers[ gray <  (thresh - delta) ] = 1  
    markers[ gray >  (thresh + delta) ] = 2  
    markers[gray < 0.1] = 2 
    # (d) Run Random Walker
    rw_labels = seg.random_walker(gray, markers, beta=90, mode='bf')
    
    pseudo = (rw_labels == 1).astype(np.uint8)
    # (e) Morphological cleanup: fill small holes, remove small objects
    pseudo = morph.remove_small_holes(pseudo.astype(bool), area_threshold=32*32).astype(np.uint8)
    pseudo = morph.remove_small_objects(pseudo.astype(bool), min_size=64).astype(np.uint8)

    return torch.from_numpy(pseudo).unsqueeze(0).float()


def data_transform():
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)

    # --- CROP ONLY (used before RW) ---
    crop_transform_global = v2.Compose([
        v2.RandomResizedCrop(256, scale=(0.4, 1.0), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToTensor()
    ])
    
    crop_transform_local = v2.Compose([
        v2.RandomResizedCrop(128, scale=(0.4, 1), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToTensor()
    ])

    # --- COLOR ONLY (used after pseudo) ---
    color_transform_global = v2.Compose([
        v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        v2.RandomGrayscale(p=0.2),
        v2.RandomApply([v2.GaussianBlur(kernel_size=21, sigma=(0.1, 2.0))], p=0.5),
        #v2.RandomSolarize(threshold=0.5, p=0.2),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    
    color_transform_local = v2.Compose([
        v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        v2.RandomGrayscale(p=0.2),
        v2.RandomApply([v2.GaussianBlur(kernel_size=21, sigma=(0.1, 2.0))], p=0.5),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    return DinoMultiCropTransform(
        crop_transform_global, color_transform_global,
        crop_transform_local, color_transform_local,
        n_global=2, n_local=4
    )

class DinoMultiCropTransform:
    def __init__(self, crop_transform_global, color_transform_global,
                 crop_transform_local, color_transform_local,
                 n_global=2, n_local=4):
        
        self.crop_global = crop_transform_global
        self.color_global = color_transform_global
        self.crop_local = crop_transform_local
        self.color_local = color_transform_local
        self.n_global = n_global
        self.n_local = n_local

        print(f"--- DinoMultiCropTransform Initialized ---")
        print(f"n_global: {self.n_global}")
        print(f"n_local : {self.n_local}")
        print(f"crop_transform_local: {self.crop_local}")
        print(f"crop_transform_global: {self.crop_global}") 
        print(f"color_transform_local: {self.color_local}")
        print(f"color_transform_global: {self.color_global}")
        print(f"----------------------------------------")

    def __call__(self, img,real_mask,pseudo_mask):
        student_crops, teacher_crops,real_mask_crops = [],[], []
        pseudo_masks = []

        # global crops for teacher
        for _ in range(self.n_global):
            cropped = self.crop_global(img)
            transformed = self.color_global(cropped)
            teacher_crops.append(transformed)

        # global crops for student + pseudo masks + real mask 
        for _ in range(self.n_global):

            i, j, h, w = v2.RandomResizedCrop.get_params(img, scale=(0.4, 1.0), ratio=(3.0/4.0, 4.0/3.0))
            pseudo_mask_512 = TF.resize(pseudo_mask, size=(512, 512), interpolation=TF.InterpolationMode.NEAREST)
            crop_img = TF.resized_crop(img, i, j, h, w, size=(256, 256), antialias=True)
            crop_mask = TF.resized_crop(real_mask, i, j, h, w, size=(256, 256), interpolation=TF.InterpolationMode.NEAREST)
            crop_pseudo = TF.resized_crop(pseudo_mask_512, i, j, h, w, size=(256, 256), interpolation=TF.InterpolationMode.NEAREST)

            if torch.rand(1) < 0.5:
                crop_img = TF.hflip(crop_img)
                crop_mask = TF.hflip(crop_mask)
                crop_pseudo = TF.hflip(crop_pseudo)

            transformed_img = self.color_global(crop_img) # Color transform sadece görüntüye!
            student_crops.append(transformed_img)
            #pseudo_mask = random_walker_pseudo_mask(crop_img)  # [1, H, W], 0 or 1
            pseudo_masks.append(crop_pseudo)
            real_mask_crops.append(crop_mask)

        # local crops for student
        for _ in range(self.n_local):
            cropped = self.crop_local(img)
            transformed = self.color_local(cropped)
            student_crops.append(transformed)

        return img, real_mask_crops,student_crops, teacher_crops, pseudo_masks  # 

def loader(op,mode,sslmode,batch_size,num_workers,image_size,cutout_pr,cutout_box,shuffle,split_ratio,data):

    if data=='isic_2018_1':
        foldernamepath="isic_2018_1/"
        imageext="/*.jpg"
        maskext="/*.png"
    elif data == 'kvasir_1':
        foldernamepath="kvasir_1/"
        imageext="/*.jpg"
        maskext="/*.jpg"
    elif data == 'ham_1':
        foldernamepath="HAM10000_1/"
        imageext="/*.jpg"
        maskext="/*.png"
    elif data == 'PH2Dataset':
        foldernamepath="PH2Dataset/"
        imageext="/*.jpeg"
        maskext="/*.jpeg"
    elif data == 'isic_2016_1':
        foldernamepath="isic_2016_1/"
        imageext="/*.jpg"
        maskext="/*.png"

    if op =="train":
        train_im_path   = os.environ["ML_DATA_ROOT"]+foldernamepath+"train/images"   
        train_mask_path = os.environ["ML_DATA_ROOT"]+foldernamepath+"train/masks"
        train_pmask_path = os.environ["ML_DATA_ROOT"]+foldernamepath+"train/pmasks"
        
        train_im_path   = sorted(glob(train_im_path+imageext))
        train_mask_path = sorted(glob(train_mask_path+maskext))
        train_pmask_path = sorted(glob(train_pmask_path+maskext))

    elif op == "validation":
        test_im_path    = os.environ["ML_DATA_ROOT"]+foldernamepath+"val/images"
        test_mask_path  = os.environ["ML_DATA_ROOT"]+foldernamepath+"val/masks"
        test_pmask_path = os.environ["ML_DATA_ROOT"]+foldernamepath+"val/pmasks"

        test_im_path    = sorted(glob(test_im_path+imageext))
        test_mask_path  = sorted(glob(test_mask_path+maskext))
        test_pmask_path = sorted(glob(test_pmask_path+maskext))

    else :
        test_im_path    = os.environ["ML_DATA_ROOT"]+foldernamepath+"test/images"
        test_mask_path  = os.environ["ML_DATA_ROOT"]+foldernamepath+"test/masks"
        test_pmask_path = os.environ["ML_DATA_ROOT"]+foldernamepath+"test/pmasks"

        test_im_path    = sorted(glob(test_im_path+imageext))
        test_mask_path  = sorted(glob(test_mask_path+maskext))
        test_pmask_path = sorted(glob(test_pmask_path+maskext))

    transformations = data_transform()

    if torch.cuda.is_available():
        if op == "train":
            data_train  = dataset(train_im_path,train_mask_path,train_pmask_path,cutout_pr,cutout_box, transformations,mode)
        else:
            data_test   = dataset(test_im_path, test_mask_path,test_pmask_path,cutout_pr,cutout_box, transformations,mode)

    elif op == "train":  #train for debug in local
        data_train  = dataset(train_im_path[5:15],train_mask_path[5:15],train_pmask_path[5:15],cutout_pr,cutout_box, transformations,mode)

    else:  #test in local
        data_test   = dataset(test_im_path[5:15], test_mask_path[5:15], test_pmask_path[5:15],cutout_pr,cutout_box, transformations,mode)

    if op == "train":
        train_loader = DataLoader(
            dataset     = data_train,
            batch_size  = batch_size,
            shuffle     = shuffle,
            num_workers = num_workers,
            persistent_workers=True
            )
        return train_loader
    
    else :
        test_loader = DataLoader(
            dataset     = data_test,
            batch_size  = batch_size,
            shuffle     = shuffle,
            num_workers = num_workers,
        )
    
        return test_loader


#loader()