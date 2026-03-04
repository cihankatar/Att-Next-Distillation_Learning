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

def data_transform():
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)

    # --- CROP ONLY (used before RW) ---
    crop_transform_global = v2.Compose([
        v2.RandomResizedCrop(256, scale=(0.4, 1.0), ratio=(3/4, 4/3), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToTensor()
    ])

    crop_transform_local = v2.Compose([
        v2.RandomResizedCrop(128, scale=(0.1, 0.4), ratio=(3/4, 4/3), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToTensor()
    ])

    # --- COLOR ONLY (used after pseudo) ---
    color_transform_global = v2.Compose([
        v2.RandomApply([v2.ColorJitter(0.3, 0.3, 0.3, 0.0)], p=0.8),
        v2.RandomGrayscale(p=0.2),
        v2.RandomApply([v2.GaussianBlur(kernel_size=21, sigma=(0.1, 2.0))], p=0.3),
        #v2.RandomSolarize(threshold=0.5, p=0.2),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    
    color_transform_local = v2.Compose([
        v2.RandomApply([v2.ColorJitter(0.3, 0.3, 0.3, 0.0)], p=0.8),
        v2.RandomGrayscale(p=0.2),
        v2.RandomApply([v2.GaussianBlur(kernel_size=11, sigma=(0.1, 2.0))], p=0.5),
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

        # v2.RandomErasing'i buradan kaldırdık çünkü koordinatları içeride manuel yöneteceğiz
        self.erasing_p = 0.0
        self.erasing_scale = (0.05, 0.1)
        self.erasing_ratio = (0.3, 3.3)
        
        print(f"--- DinoMultiCropTransform Initialized ---")
        print(f"n_global: {self.n_global}")
        print(f"n_local : {self.n_local}")
        print(f"----------------------------------------")

    def __call__(self, img, real_mask, pseudo_mask):
        student_crops, teacher_crops, real_mask_crops = [], [], []
        pseudo_masks = []
        
        # 1. Pseudo Mask hizalaması (Döngü dışına aldık, performans için)
        pseudo_mask_512 = TF.resize(pseudo_mask, size=(512, 512), interpolation=TF.InterpolationMode.NEAREST)
        
        # 2. Global crops for teacher (Maskesiz)
        for _ in range(self.n_global):
            cropped = self.crop_global(img)
            transformed = self.color_global(cropped)
            teacher_crops.append(transformed)

        # 3. Global crops for student + pseudo masks + real mask 
        for _ in range(self.n_global):
            i, j, h, w = v2.RandomResizedCrop.get_params(img, scale=(0.4, 1.0), ratio=(3.0/4.0, 4.0/3.0))
            
            crop_img = TF.resized_crop(img, i, j, h, w, size=(256, 256), antialias=True)
            crop_mask = TF.resized_crop(real_mask, i, j, h, w, size=(256, 256), interpolation=TF.InterpolationMode.NEAREST)
            crop_pseudo = TF.resized_crop(pseudo_mask_512, i, j, h, w, size=(256, 256), interpolation=TF.InterpolationMode.NEAREST)

            if torch.rand(1) < 0.5:
                crop_img = TF.hflip(crop_img)
                crop_mask = TF.hflip(crop_mask)
                crop_pseudo = TF.hflip(crop_pseudo)

            # Tensor Dönüşümü (Erase ve Color işlemleri için zorunludur)
            if not isinstance(crop_img, torch.Tensor):
                crop_img = TF.to_tensor(crop_img)
            if not isinstance(crop_mask, torch.Tensor):
                crop_mask = TF.to_tensor(crop_mask)
            if not isinstance(crop_pseudo, torch.Tensor):
                crop_pseudo = TF.to_tensor(crop_pseudo)

            # Sadece görüntüye renk transformu
            transformed_img = self.color_global(crop_img) 
            


            # Listelere Ekleme
            student_crops.append(transformed_img)
            real_mask_crops.append(crop_mask)
            pseudo_masks.append(crop_pseudo)

        # 4. Local crops for student
        for _ in range(self.n_local):
            cropped = self.crop_local(img)
            transformed = self.color_local(cropped)
            student_crops.append(transformed)

        return img, real_mask_crops, student_crops, teacher_crops, pseudo_masks
    
    
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