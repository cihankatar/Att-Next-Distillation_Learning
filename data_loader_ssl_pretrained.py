from torch.utils.data import DataLoader

from data.Custom_Dataset_ssl_pretrained import dataset
from utils.Test_Train_Split import ssl_data_split
from glob import glob
from torchvision.transforms import v2 
import os
import numpy as np
import torch


def data_transform(train,image_size):
        
        if train:
        
            transformations = v2.Compose([  v2.Resize([image_size,image_size],antialias=True),                                           
                                        v2.RandomHorizontalFlip(p=0.5),
                                        v2.RandomVerticalFlip(p=0.5),
                                        v2.RandomRotation(degrees=(0, 90)),
                                        #v2.RandomAdjustSharpness(sharpness_factor=10, p=0.),
                                        #v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
                                        #v2.RandomPerspective(distortion_scale=0.5, p=0.5),
                                        #v2.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.75, 0.75)),
                                        #v2.RandomPhotometricDistort(p=0.3),
                                        #v2.Normalize(mean=(0.400, 0.485, 0.456, 0.406), std=(0,222, 0.229, 0.224, 0.225))                              
                                    ])
        else:
            transformations = v2.Compose([  v2.Resize([image_size,image_size],antialias=True),
                            #v2.RandomHorizontalFlip(p=0.5),
                            #v2.RandomVerticalFlip(p=0.5),
                            #v2.RandomRotation(degrees=(0, 90)),
                            #v2.Normalize(mean=(0.400, 0.485, 0.456, 0.406), std=(0,222, 0.229, 0.224, 0.225)),                                
                            ])
            
        return transformations


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
        
        train_im_path   = sorted(glob(train_im_path+imageext))
        train_mask_path = sorted(glob(train_mask_path+maskext))
    
    elif op == "validation":
        test_im_path    = os.environ["ML_DATA_ROOT"]+foldernamepath+"val/images"
        test_mask_path  = os.environ["ML_DATA_ROOT"]+foldernamepath+"val/masks"
        test_im_path    = sorted(glob(test_im_path+imageext))
        test_mask_path  = sorted(glob(test_mask_path+maskext))

    else :
        test_im_path    = os.environ["ML_DATA_ROOT"]+foldernamepath+"test/images"
        test_mask_path  = os.environ["ML_DATA_ROOT"]+foldernamepath+"test/masks"
        test_im_path    = sorted(glob(test_im_path+imageext))
        test_mask_path  = sorted(glob(test_mask_path+maskext))

    transformations = data_transform()

    if torch.cuda.is_available():
        if op == "train":
            data_train  = dataset(train_im_path,train_mask_path,cutout_pr,cutout_box, transformations,mode)
        else:
            data_test   = dataset(test_im_path, test_mask_path,cutout_pr,cutout_box, transformations,mode)

    elif op == "train":  #train for debug in local
        data_train  = dataset(train_im_path[10:20],train_mask_path[10:20],cutout_pr,cutout_box, transformations,mode)

    else:  #test in local
        data_test   = dataset(test_im_path[10:20], test_mask_path[10:20],cutout_pr,cutout_box, transformations,mode)

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