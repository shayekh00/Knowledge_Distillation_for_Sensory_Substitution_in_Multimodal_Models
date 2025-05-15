from PIL import Image
import pandas as pd
import torchvision.transforms.functional as TF
import random
import torchvision.transforms as transforms
import os
from torch.utils.data import Dataset
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

def remove_substring_from_path(path, substring="SUNRGBD"):
    index = path.find(substring)
    if index != -1:
        path = path[:index] + path[index + len(substring):]
    
    return path

class CustomSUNRGBDDataset(Dataset):
    def __init__(self,ROOT_DATA_DIR, csv_file_name, transform=None, augmentation=True, subset_percentage=None):
        self.csv_file_path = os.path.join(ROOT_DATA_DIR, "SUNRGBD/csv_data", csv_file_name)
        self.df = pd.read_csv(self.csv_file_path)
        self.transform = transform
        self.augmentation = augmentation

        if subset_percentage is not None:
            total_samples = len(self.df)
            subset_size = int(total_samples * subset_percentage)
            self.df = self.df.iloc[:subset_size]
        
        self.dataset_directory = os.path.join(ROOT_DATA_DIR, "SUNRGBD")
        # self.dataset_directory = 'dataset/output_path/SUNRGBD/'

        # Define your augmentations using albumentations
        self.augmentation_pipeline = A.Compose([
            A.HorizontalFlip(p=0.5), 
            A.RandomBrightnessContrast(p=0.2), 
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, p=0.5), 
            A.GaussianBlur(p=0.2),
            A.CoarseDropout(p=0.5, max_holes=8, max_height=16, max_width=16, min_holes=1),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # Normalizing
            # ToTensorV2()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        #Directory processing
        img_name_rgb = self.df.iloc[idx, 3]
        img_name_depth = self.df.iloc[idx, 4]
        directory = self.dataset_directory
        rgb_image_path = os.path.join(directory, img_name_rgb) 
        depth_image_path = os.path.join(directory, img_name_depth) 
        rgb_image_path = remove_substring_from_path(rgb_image_path).replace("\\", "/")
        depth_image_path = remove_substring_from_path(depth_image_path).replace("\\", "/")


        rgb_image = np.array(Image.open(rgb_image_path))

        depth_image = Image.open(depth_image_path)
        # Repeat the single channel 3 times
        depth_image_array = np.array(depth_image)
        depth_image_3channel_array = np.stack([depth_image_array] * 3, axis=-1)

        question = self.df.iloc[idx, 1]
        answer = self.df.iloc[idx, 2]


        if self.augmentation:
            # Apply the same augmentation to both RGB and Depth images
            augmented = self.augmentation_pipeline(image=rgb_image, mask=depth_image_3channel_array)
            rgb_image = augmented['image']
            depth_image = augmented['mask']

            # Convert back to PIL Image
            rgb_image = Image.fromarray(rgb_image.astype(np.uint8))
            depth_image = Image.fromarray(depth_image.astype(np.uint8))
        else:
            # Convert to tensor without augmentation
            rgb_image = Image.open(rgb_image_path)
            depth_image = Image.fromarray(depth_image_3channel_array.astype(np.uint8))
            
        return question, answer, rgb_image, depth_image ,idx
