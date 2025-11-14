import os
import math
import random
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import cv2
from PIL import Image

import utils

ALLMASKTYPES = ['single_bbox', 'bbox', 'free_form']

class DomainTransferDataset(Dataset):
    def __init__(self, opt):
        super(DomainTransferDataset, self).__init__()
        self.opt = opt
        self.imglist_A = utils.get_files(opt.baseroot_A)
        self.imglist_B = utils.get_files(opt.baseroot_B)
        self.len_A = len(self.imglist_A)
        self.len_B = len(self.imglist_B)
    
    def imgcrop(self, img):
        H, W = img.shape
        # scaled size should be greater than opts.crop_size
        if H < W:
            if H < self.opt.crop_size:
                H_out = self.opt.crop_size
                W_out = int(math.floor(W * float(H_out) / float(H)))
                img = cv2.resize(img, (W_out, H_out))
        else: # W_out < H_out
            if W < self.opt.crop_size:
                W_out = self.opt.crop_size
                H_out = int(math.floor(H * float(W_out) / float(W)))
                img = cv2.resize(img, (W_out, H_out))
        # randomly crop
        rand_h = random.randint(0, max(0, H - self.opt.imgsize))
        rand_w = random.randint(0, max(0, W - self.opt.imgsize))
        img = img[rand_h:rand_h + self.opt.imgsize, rand_w:rand_w + self.opt.imgsize, :]
        return img

    def __getitem__(self, index):

        ## Image A
        random_A = random.randint(0, self.len_A - 1)
        imgpath_A = self.imglist_A[random_A]
        img_A = cv2.imread(imgpath_A, cv2.IMREAD_GRAY)
        # image cropping
        img_A = self.imgcrop(img_A)
        
        ## Image B
        random_B = random.randint(0, self.len_B - 1)
        imgpath_B = self.imglist_B[random_B]
        img_B = cv2.imread(imgpath_B, cv2.IMREAD_GRAY)
        # image cropping
        img_B = self.imgcrop(img_B)

        # To tensor (grayscale)
        img_A = torch.from_numpy(img_A.astype(np.float32) / 255.0).unsqueeze(0).contiguous()
        img_B = torch.from_numpy(img_B.astype(np.float32) / 255.0).unsqueeze(0).contiguous()

        return img_A, img_B
    
    def __len__(self):
        return min(self.len_A, self.len_B)

# Note: InpaintDataset, InpaintDataset_val, and ValidationSet_with_Known_Mask classes
# are kept for backward compatibility but are no longer used in bubble mode training.
# Random mask generation functions have been removed as they are not needed with pre-generated masks.

class InpaintDataset(Dataset):
    """Legacy dataset class - use BubbleInpaintDataset for bubble mode instead."""
    def __init__(self, opt):
        print("Warning: InpaintDataset is deprecated. Use BubbleInpaintDataset with bubble mode instead.")
        assert opt.mask_type in ALLMASKTYPES
        self.opt = opt
        self.imglist = utils.get_jpgs(opt.baseroot)

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):
        raise NotImplementedError("InpaintDataset is deprecated. Use BubbleInpaintDataset with bubble mode instead.")

class InpaintDataset_val(Dataset):
    """Legacy validation dataset class - use BubbleInpaintDataset for bubble mode instead."""
    def __init__(self, opt):
        print("Warning: InpaintDataset_val is deprecated. Use BubbleInpaintDataset with bubble mode instead.")
        assert opt.mask_type in ALLMASKTYPES
        self.opt = opt
        self.imglist = utils.get_jpgs(opt.baseroot)

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):
        raise NotImplementedError("InpaintDataset_val is deprecated. Use BubbleInpaintDataset with bubble mode instead.")

class ValidationSet_with_Known_Mask(Dataset):
    """Legacy validation dataset class - use BubbleInpaintDataset for bubble mode instead."""
    def __init__(self, opt):
        print("Warning: ValidationSet_with_Known_Mask is deprecated. Use BubbleInpaintDataset with bubble mode instead.")
        self.opt = opt
        self.namelist = utils.get_jpgs(opt.baseroot)

    def __len__(self):
        return len(self.namelist)

    def __getitem__(self, index):
        raise NotImplementedError("ValidationSet_with_Known_Mask is deprecated. Use BubbleInpaintDataset with bubble mode instead.")
