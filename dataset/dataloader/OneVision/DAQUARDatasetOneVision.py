from PIL import Image
import pandas as pd
import torchvision.transforms as transforms
import os
from torch.utils.data import Dataset
import numpy as np
from scipy.ndimage import convolve

def remove_substring_from_path(path, substring="SUNRGBD"):
    index = path.find(substring)
    if index != -1:
        path = path[:index] + path[index + len(substring):]
    return path

class DAQUARDatasetOneVision(Dataset):
    def __init__(self, root_data_dir, csv_file_name, augmentation=True, subset_percentage=None):
        self.csv_file_path = os.path.join(root_data_dir, "DAQUAR_Dataset/dataset", csv_file_name)
        self.df = pd.read_csv(self.csv_file_path)
        self.augmentation = augmentation
        self.dataset_directory = os.path.join(root_data_dir, "DAQUAR_Dataset/dataset")

        self.rgb_augmentations = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])    
        # self.model_config = model_config
        # self.processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf") 
        
        # Preprocessing for depth image (normalization)
        self.depth_preprocessing = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])        
        
        if subset_percentage is not None:
            total_samples = len(self.df)
            subset_size = int(total_samples * subset_percentage)
            self.df = self.df.iloc[:subset_size]

        
        

    def __len__(self):
        return len(self.df)


    def convert_depth_image_into_3D(self, depth_image_path):
        """
        Loads a depth image, applies Prewitt filtering to compute gradient
        magnitude and direction, then returns a 3-channel PIL image:
        (1) normalized depth, (2) gradient magnitude, (3) gradient direction
        """
        # Define Prewitt kernels
        Kx = np.array([[-1,  0,  1],
                    [-1,  0,  1],
                    [-1,  0,  1]], dtype=np.float32)
        Ky = np.array([[-1, -1, -1],
                    [ 0,  0,  0],
                    [ 1,  1,  1]], dtype=np.float32)

        # Helper function to safely normalize an array to [0, 255]
        def safe_normalize(arr):
            a_min, a_max = arr.min(), arr.max()
            if a_max == a_min:
                a_max = a_min + 1e-6
            return 255.0 * (arr - a_min) / (a_max - a_min)

        # 1. Load depth image and convert to float
        depth_image = Image.open(depth_image_path).convert("I")  # "I" for 32-bit grayscale
        depth_array = np.array(depth_image, dtype=np.float32)

        # 2. Normalize depth to [0, 255]
        depth_min = depth_array.min()
        depth_max = depth_array.max()
        if depth_max == depth_min:
            depth_max = depth_min + 1e-6
        depth_norm = (255.0 * (depth_array - depth_min) / (depth_max - depth_min)).astype(np.uint8)

        # 3. Apply Prewitt kernels
        Gx = convolve(depth_norm.astype(np.float32), Kx, mode='reflect')
        Gy = convolve(depth_norm.astype(np.float32), Ky, mode='reflect')

        # 4. Compute gradient magnitude and direction
        Gm = np.sqrt(Gx**2 + Gy**2)
        Gtheta = np.arctan2(Gy, Gx)  # range: [-pi, pi]

        # 5. Normalize magnitude and direction to [0, 255]
        Gm_norm = safe_normalize(Gm).astype(np.uint8)
        Gtheta_norm = safe_normalize(Gtheta).astype(np.uint8)

        # 6. Stack into a 3-channel image
        depth_3channel = np.dstack([depth_norm, Gm_norm, Gtheta_norm])
        depth_3channel_pil = Image.fromarray(depth_3channel, mode="RGB")

        return depth_3channel_pil


    # def convert_depth_image(self, depth_image_path):
    #     # Load depth image
    #     depth_image = Image.open(depth_image_path)
    #     depth_image_array = np.array(depth_image)

    #     depth_min = depth_image_array.min()
    #     depth_max = depth_image_array.max()

    #     depth_image_normalized = (255 * (depth_image_array - depth_min) / (depth_max - depth_min)).astype(np.uint8)

    #     depth_image_3channel_array = np.stack([depth_image_normalized] * 3, axis=-1)

    #     depth_image_pil = Image.fromarray(depth_image_3channel_array)

    #     return depth_image_pil

    def get_image_paths(self, img_name_rgb, img_name_depth):
        """
        Helper function to build and clean the paths for RGB and depth images.
        """
        directory = self.dataset_directory
        
        img_name_rgb = img_name_rgb + ".png"
        img_name_depth = img_name_depth + "_depth.png"
        # Build full paths
        rgb_image_path = os.path.join(directory,"images", img_name_rgb)
        depth_image_path = os.path.join(directory,"depth", img_name_depth)

        # Clean up the paths (e.g., remove substrings and fix slashes)
        # rgb_image_path = remove_substring_from_path(rgb_image_path).replace("\\", "/")
        # depth_image_path = remove_substring_from_path(depth_image_path).replace("\\", "/")

        return rgb_image_path, depth_image_path
    
    # def __getitem__(self, idx):
    #     img_name_rgb = self.df.iloc[idx, 3]
    #     img_name_depth = self.df.iloc[idx, 4]

    #     rgb_image_path, depth_image_path = self.get_image_paths(img_name_rgb, img_name_depth)

    #     rgb_image = Image.open(rgb_image_path)
    #     rgb_image_np = np.array(rgb_image)

    #     # depth_image = self.convert_depth_image(depth_image_path)
    #     depth_image = self.convert_depth_image_into_3D(depth_image_path)
    #     depth_image_array = np.array(depth_image)

    #     question = self.df.iloc[idx, 1]
    #     answer = self.df.iloc[idx, 2]

    #     return question, answer, rgb_image_np, depth_image_array, idx 


    # def __getitem__(self, idx):
    #     img_name_rgb = self.df.iloc[idx, 3]
    #     img_name_depth = self.df.iloc[idx, 4]

    #     rgb_image_path, depth_image_path = self.get_image_paths(img_name_rgb, img_name_depth)

    #     # Load images
    #     rgb_image = Image.open(rgb_image_path).convert('RGB')
    #     depth_image = self.convert_depth_image_into_3D(depth_image_path)

    #     # Apply augmentations if enabled
    #     if self.augmentation:
    #         rgb_image = self.augmentations(rgb_image)

    #     question = self.df.iloc[idx, 1]
    #     answer = self.df.iloc[idx, 2]

    #     return question, answer, rgb_image, depth_image, idx

    def __getitem__(self, idx):
        img_name_rgb = self.df.iloc[idx, 2]
        img_name_depth = self.df.iloc[idx, 2]


        rgb_image_path, depth_image_path = self.get_image_paths(img_name_rgb, img_name_depth)

        # Load images
        rgb_image = Image.open(rgb_image_path)
        rgb_image_np = np.array(rgb_image)

        depth_image = self.convert_depth_image_into_3D(depth_image_path)
        depth_image_array = np.array(depth_image)

        # Apply augmentations only during training
        if self.augmentation:
            # Apply augmentations on RGB image
            rgb_image = self.rgb_augmentations(rgb_image)
        else:
            # Just apply the preprocessing pipeline on RGB image
            rgb_image = self.rgb_augmentations(rgb_image)

        # Apply depth preprocessing (no augmentation)
        depth_image = self.depth_preprocessing(depth_image)

        question = self.df.iloc[idx, 1]
        answer = self.df.iloc[idx, 2]

        return question, answer, rgb_image_np, depth_image_array, idx