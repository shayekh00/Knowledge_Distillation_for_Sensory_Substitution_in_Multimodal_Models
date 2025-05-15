from PIL import Image
import pandas as pd
import torchvision.transforms.functional as TF
import random
import torchvision.transforms as transforms
import os
from torch.utils.data import Dataset
import numpy as np
import albumentations as A
# from albumentations.pytorch import ToTensorV2
from llava.mm_utils import tokenizer_image_token
import copy
import torch
from transformers import AutoProcessor

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

def remove_substring_from_path(path, substring="SUNRGBD"):
    index = path.find(substring)
    if index != -1:
        path = path[:index] + path[index + len(substring):]
    return path

class CustomSUNRGBDDatasetOneVision1DDepth(Dataset):
    def __init__(self, root_data_dir, csv_file_name, model_config=None, augmentation=True, subset_percentage=None):
        self.csv_file_path = os.path.join(root_data_dir, "SUNRGBD/csv_data", csv_file_name)
        self.df = pd.read_csv(self.csv_file_path)
        self.augmentation = augmentation
        # self.model_config = model_config
        # self.processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf") 
        
        if subset_percentage is not None:
            total_samples = len(self.df)
            subset_size = int(total_samples * subset_percentage)
            self.df = self.df.iloc[:subset_size]
        
        self.dataset_directory = os.path.join(root_data_dir, "SUNRGBD")
        
        self.augmentation_pipeline = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, p=0.5),
            A.GaussianBlur(p=0.2),
            A.CoarseDropout(p=0.5, max_holes=8, max_height=16, max_width=16, min_holes=1),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return len(self.df)

    def convert_depth_image(self, depth_image_path):
        # Load depth image
        depth_image = Image.open(depth_image_path)
        depth_image_array = np.array(depth_image)

        depth_min = depth_image_array.min()
        depth_max = depth_image_array.max()

        depth_image_normalized = (255 * (depth_image_array - depth_min) / (depth_max - depth_min)).astype(np.uint8)

        depth_image_3channel_array = np.stack([depth_image_normalized] * 3, axis=-1)

        depth_image_pil = Image.fromarray(depth_image_3channel_array)

        return depth_image_pil

    def get_image_paths(self, img_name_rgb, img_name_depth):
        """
        Helper function to build and clean the paths for RGB and depth images.
        """
        directory = self.dataset_directory
        
        # Build full paths
        rgb_image_path = os.path.join(directory, img_name_rgb)
        depth_image_path = os.path.join(directory, img_name_depth)

        # Clean up the paths (e.g., remove substrings and fix slashes)
        rgb_image_path = remove_substring_from_path(rgb_image_path).replace("\\", "/")
        depth_image_path = remove_substring_from_path(depth_image_path).replace("\\", "/")

        return rgb_image_path, depth_image_path
    
    def __getitem__(self, idx):
        img_name_rgb = self.df.iloc[idx, 3]
        img_name_depth = self.df.iloc[idx, 4]

        rgb_image_path, depth_image_path = self.get_image_paths(img_name_rgb, img_name_depth)

        rgb_image = Image.open(rgb_image_path)
        rgb_image_np = np.array(rgb_image)

        depth_image = Image.open(depth_image_path)
        depth_image_array = np.array(depth_image)

        question = self.df.iloc[idx, 1]
        answer = self.df.iloc[idx, 2]

        return question, answer, rgb_image_np, depth_image_array, idx 

        # if self.augmentation:
        #     augmented = self.augmentation_pipeline(image=rgb_image, mask=depth_image_3channel_array)
        #     rgb_image = augmented['image']
        #     depth_image = augmented['mask']
        # else:
        #     rgb_image = Image.open(rgb_image_path)
        #     rgb_image = Image.fromarray(depth_image_3channel_array.astype(np.uint8))

        # # Process images for OneVision model
        # rgb_image_processed = process_images([rgb_image], self.image_processor, self.model.config) 
        # depth_image_processed = process_images([rgb_image], self.image_processor, self.model.config) 

        # # Tokenize input
        # prompt_question = self.get_prompt_question(question)
        # input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").squeeze()

        # # Tokenize answer (labels)
        # labels = self.tokenizer(answer, return_tensors="pt", padding="max_length", truncation=True).input_ids.squeeze()

        # return question, answer, rgb_image, depth_image_3channel_array, idx
        

