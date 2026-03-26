import os 
import sys

import cv2

from torch.utils.data import Dataset


class VOCDataset(Dataset):
    def __init__(self, data_path, data_type="train", transform=None):
        super().__init__()
        
        self.data_path = data_path
        self.data_type = data_type
        self.transform = transform
        self.data = self._load_data(self.data_type)


    def _load_data(self, data_type):
        file_path = os.path.join(self.data_path, "ImageSets", "Segmentation", f"{data_type}.txt")

        with open(file_path, 'r') as f:
            data = [line.strip() for line in f.readlines()]
        
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_num = self.data[idx]

        img_path = os.path.join(self.data_path, "JPEGImages", f"{img_num}.jpg")
        mask_path = os.path.join(self.data_path, "SegmentationClass", f"{img_num}.png")
        
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path)

        return image, mask