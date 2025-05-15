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

class CustomSUNRGBDDatasetColor(Dataset):
    def __init__(self,ROOT_DATA_DIR, csv_file_name, transform=None, augmentation=True, subset_percentage=None):
        self.csv_file_path = os.path.join(ROOT_DATA_DIR, "SUNRGBD/csv_data/individual_datasets/", csv_file_name)
        self.df = pd.read_csv(self.csv_file_path)
        self.transform = transform
        self.augmentation = augmentation

        if subset_percentage is not None:
            total_samples = len(self.df)
            subset_size = int(total_samples * subset_percentage)
            self.df = self.df.iloc[:subset_size]
        
        self.dataset_directory = os.path.join(ROOT_DATA_DIR, "SUNRGBD")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):


        #Directory processing
        img_name_rgb = self.df.iloc[idx, 2]
        img_name_depth = self.df.iloc[idx, 3]
        directory = self.dataset_directory
        rgb_image_path = os.path.join(directory, img_name_rgb) 
        depth_image_path = os.path.join(directory, img_name_depth) 
        rgb_image_path = remove_substring_from_path(rgb_image_path).replace("\\", "/")
        depth_image_path = remove_substring_from_path(depth_image_path).replace("\\", "/")


        rgb_image = Image.open(rgb_image_path)

        depth_image = Image.open(depth_image_path)
        # Repeat the single channel 3 times
        depth_image_array = np.array(depth_image)
        depth_image_3channel_array = np.stack([depth_image_array] * 3, axis=-1)
        depth_image = Image.fromarray(depth_image_3channel_array.astype(np.uint8))

        question = self.df.iloc[idx, 0]
        answer = self.df.iloc[idx, 1]

        # if self.transform:
        #     if self.augmentation:
        #         # Random horizontal flip
        #         if random.random() > 0.5:
        #             rgb_image = TF.hflip(rgb_image)
        #             depth_image = TF.hflip(depth_image)

        #     rgb_image = self.transform(rgb_image)
        #     depth_image = transforms.ToTensor()(depth_image)
        #     # depth_image = depth_image.repeat(3, 1, 1)  # Repeat the single channel 3 times
        #     depth_image = depth_image.float()  # Convert to float
        #     depth_image = depth_image / 255.0  # Normalize to [0,1]
        #     depth_image = TF.to_pil_image(depth_image)
        #     depth_image = self.transform(depth_image)  # Normalize further

        #     if self.augmentation:
        #         # Random rotation
        #         angle = random.randint(-10, 10)
        #         rgb_image = TF.rotate(rgb_image, angle)
        #         depth_image = TF.rotate(depth_image, angle)

            
        return question, answer, rgb_image, depth_image ,idx
