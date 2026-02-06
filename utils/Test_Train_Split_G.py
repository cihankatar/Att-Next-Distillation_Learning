import os
import shutil
import random
from PIL import Image

# Paths
image_path = "/Users/input/data/ckatar/isic_2016_1/images"
mask_path = "/Users/input/data/ckatar/isic_2016_1/masks"
output_base = "/Users/input/data/ckatar/isic_2016_1/split_data"

# Splits
splits = ['train', 'test', 'val']

# Create output directories
for split in splits:
    os.makedirs(os.path.join(output_base, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_base, split, 'masks'), exist_ok=True)

# List all images
images = [f for f in os.listdir(image_path) if f.endswith('.jpg')]

# Randomize and split
random.shuffle(images)
train_split, test_split, val_split = 160, 20, 20
train_images = images[:train_split]
test_images = images[train_split:train_split + test_split]
val_images = images[train_split + test_split:]

# Helper function to resize and copy files
def resize_and_copy_files(image_list, split):
    for image_name in image_list:
        # Corresponding mask name
        mask_name = image_name.replace('.jpg', '_Segmentation.png')
        
        # Source paths
        image_src = os.path.join(image_path, image_name)
        mask_src = os.path.join(mask_path, mask_name)
        
        # Destination paths
        image_dst = os.path.join(output_base, split, 'images', image_name)
        mask_dst = os.path.join(output_base, split, 'masks', mask_name)
        
        # Resize and copy files
        if os.path.exists(image_src) and os.path.exists(mask_src):
            # Resize image
            with Image.open(image_src) as img:
                img = img.resize((256, 256))
                img.save(image_dst)

            # Resize mask
            with Image.open(mask_src) as mask:
                mask = mask.resize((256, 256))
                mask.save(mask_dst)

# Copy and resize to respective folders
resize_and_copy_files(train_images, 'train')
resize_and_copy_files(test_images, 'test')
resize_and_copy_files(val_images, 'val')

print("Data split, resized, and copied successfully!")
