import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

LABELS = [
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Consolidation", "ILD", "Infiltration", "Lung Opacity",
    "Nodule/Mass", "Other lesion", "Pleural effusion", "Pleural thickening",
    "Pneumothorax", "Pulmonary fibrosis"
]

def crop_center_blind(image, crop_percent=0.08):
    #Cắt bỏ X% viền xung quanh ảnh để loại bỏ nhiễu (chữ L, R, Portable)
    h, w, _ = image.shape
    margin_h = int(h * crop_percent) 
    margin_w = int(w * crop_percent)
    
    if margin_h * 2 >= h or margin_w * 2 >= w:
        return image
    return image[margin_h:h-margin_h, margin_w:w-margin_w]

class VinDrCXRDataset(Dataset):
    def __init__(self, csv_path, images_dir, indices=None, transform=None):
        df = pd.read_csv(csv_path)
        df['path'] = df['Image Index'].apply(lambda x: os.path.join(images_dir, x))
        df = df[df['path'].apply(os.path.exists)].reset_index(drop=True)
        
        for label in LABELS:
            df[label] = df['Finding Labels'].map(lambda x: 1.0 if label in x else 0.0)
            
        if indices is not None:
            df = df.iloc[indices].reset_index(drop=True)
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = cv2.imread(row['path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Blind Crop
        image = crop_center_blind(image, crop_percent=0.08)

        if self.transform:
            image = self.transform(image=image)['image']
            
        labels = row[LABELS].values.astype(np.float32)
        return image, torch.tensor(labels), row['Image Index']

def get_transforms(image_size=380, train=True):
    if train:
        return A.Compose([
            A.Resize(int(image_size * 1.1), int(image_size * 1.1)),
            A.RandomCrop(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.2),
            A.Affine(scale=(0.9, 1.1), translate_percent=(-0.0625, 0.0625), rotate=(-20, 20), p=0.5),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
            A.OneOf([A.GridDistortion(p=0.5), A.OpticalDistortion(distort_limit=1, p=0.5)], p=0.3),
            
            # CoarseDropout: Che ngẫu nhiên để chống model học vẹt ký tự còn sót
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, min_holes=2, min_height=16, min_width=16, fill_value=0, p=0.5),
            
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])