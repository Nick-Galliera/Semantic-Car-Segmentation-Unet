import torch
from torch.utils.data import Dataset

import numpy as np
from PIL import Image
import os
import glob


class CarSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_transform=None, mask_transform=None):
        
        self.image_dir = images_dir
        self.mask_dir = masks_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        self.images = sorted(glob.glob(os.path.join(images_dir, "*.png")))
        self.masks = sorted(glob.glob(os.path.join(masks_dir, "*.png")))

        if len(self.images) != len(self.masks):
            raise ValueError("[!] Masks number and data do not correspond")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        
        image_path = self.images[index]
        mask_path = self.masks[index]
        
        image = Image.open(image_path).convert("RGB")  
        mask = Image.open(mask_path) 

        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask
