##IMPORT 
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
##### DULLRAZOR ###
import torch
import torch.nn.functional as F

def blackhat_transform(gray_tensor, kernel_size=9):
    padding = kernel_size // 2
    kernel = torch.ones((1, 1, kernel_size, kernel_size), device=gray_tensor.device)
    dilated = F.max_pool2d(gray_tensor, kernel_size, 1, padding)
    closed = -F.max_pool2d(-dilated, kernel_size, 1, padding)
    blackhat = (closed - gray_tensor).clamp(0, 1)
    return blackhat

def patch_fill(image, mask, kernel_size=15):
    padding = kernel_size // 2
    ones = torch.ones_like(image)
    masked_input = image * (1 - mask)
    norm = F.avg_pool2d(ones * (1 - mask), kernel_size, 1, padding) + 1e-8
    smooth = F.avg_pool2d(masked_input, kernel_size, 1, padding) / norm
    return image * (1 - mask) + smooth * mask

def dullrazor(img, th=0.05):

    C, H, W = img.shape
    img = img.unsqueeze(0)  # -> [1, C, H, W]

    # Convert to grayscale using standard RGB weighting
    gray = 0.2989 * img[:, 0] + 0.5870 * img[:, 1] + 0.1140 * img[:, 2]
    gray = gray.unsqueeze(1)  # -> [1, 1, H, W]

    blackhat = blackhat_transform(gray, kernel_size=9)
    hair_mask = (blackhat > th).float()
    hair_mask_3ch = hair_mask.repeat(1, 3, 1, 1)

    clean = patch_fill(img, hair_mask_3ch)  # -> [1, 3, H, W]
    return clean.squeeze(0).clamp(0, 1)     # -> [3, H, W]
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class dataset(Dataset):
    def __init__(self, train_path, mask_path, pmask_path, cutout_pr, cutout_box, transforms, training_type):
        super().__init__()
        self.train_path = train_path
        self.mask_path = mask_path
        self.pmask_path = pmask_path
        self.tr = transforms
        self.cutout_pr = cutout_pr
        self.cutout_pad = cutout_box
        self.training_type = training_type
        self.target_size = (256, 256)

    def __len__(self):
        return len(self.train_path)

    def _resize_if_needed(self, pil_img, is_mask=False):
        if pil_img.size != self.target_size:
            if is_mask:
                pil_img = pil_img.resize(self.target_size, Image.NEAREST)
            else:
                pil_img = pil_img.resize(self.target_size, Image.BILINEAR)
        return pil_img

    def __getitem__(self, index):
        # -------- image --------
        image = Image.open(self.train_path[index]).convert("RGB")
        image = self._resize_if_needed(image, is_mask=False)
        image = np.array(image, dtype=np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))   # HWC -> CHW
        image = torch.from_numpy(image)          # [3, 256, 256]

        # -------- real mask --------
        mask = Image.open(self.mask_path[index]).convert("L")
        mask = self._resize_if_needed(mask, is_mask=True)
        mask = np.array(mask, dtype=np.float32) / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0)   # [1, 256, 256]

        # -------- pseudo mask --------
        pseudo_mask = Image.open(self.pmask_path[index]).convert("L")
        pseudo_mask = self._resize_if_needed(pseudo_mask, is_mask=True)
        pseudo_mask = np.array(pseudo_mask, dtype=np.float32) / 255.0
        pseudo_mask = torch.from_numpy(pseudo_mask).unsqueeze(0)   # [1, 256, 256]

        if self.training_type == "ssl":
            # this should return resized/aligned outputs
            image, cropped_real_mask, student_views, teacher_views, pseudo_mask = self.tr(
                image, mask, pseudo_mask
            )

            return image, self.train_path[index], cropped_real_mask, student_views, teacher_views, pseudo_mask

        return image, mask, pseudo_mask