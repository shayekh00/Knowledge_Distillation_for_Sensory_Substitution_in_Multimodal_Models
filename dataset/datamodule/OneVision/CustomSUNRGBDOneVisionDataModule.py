# #https://github.com/mahsavafaie/BZK-End2endIE/blob/e008f6086c1665a39240e53288b57e7966a15607/inferable/models/llava_next_model.py#L68
# example code for llava_next_model

import sys
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoProcessor
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from PIL import Image
import requests
import copy
import torch

import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dataset.dataloader.OneVision.CustomSUNRGBDDatasetOneVision import CustomSUNRGBDDatasetOneVision
import traceback


class CustomSUNRGBDOneVisionDataModule(pl.LightningDataModule):
    def __init__(self, root_data_dir,processor, batch_size, num_workers, subset_percentage=None, augmentation=True):
        super().__init__()
        self.root_data_dir = root_data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers  
        self.subset_percentage = subset_percentage
        self.augmentation = augmentation
        
        self.processor = processor
        # self.pretrained = "lmms-lab/llava-onevision-qwen2-7b-ov"


        

    def setup(self, stage=None):
        # self.processor = self.processor
        
        self.train_dataset = CustomSUNRGBDDatasetOneVision(
            root_data_dir=self.root_data_dir,
            csv_file_name="train_dataset.csv",
            subset_percentage=self.subset_percentage,
            augmentation=self.augmentation,
        )
        self.val_dataset = CustomSUNRGBDDatasetOneVision(
            root_data_dir=self.root_data_dir,
            csv_file_name="val_dataset.csv",
            subset_percentage=self.subset_percentage,
            augmentation=False,  # No augmentation for validation
        )
        self.test_dataset = CustomSUNRGBDDatasetOneVision(
            root_data_dir=self.root_data_dir,
            csv_file_name="test_dataset.csv",
            subset_percentage=self.subset_percentage,
            augmentation=False,  # No augmentation for validation
        )


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            shuffle=False
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            shuffle=False
        )


    def collate_fn(self, batch):
        rgb_images = []
        depth_images = []
        texts = []
        answers = []
        for example in batch:
            question, answer, rgb_image_np, depth_image_3channel_array, idx = example
            rgb_images.append(rgb_image_np)
            answers.append(answer)
            depth_images.append(depth_image_3channel_array)

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},  # Add question text
                        {"type": "image"},                  # Add the image as a type
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": answer},   # Add the assistant's answer
                    ],
                }
            ]
            text_prompt = self.processor.apply_chat_template(conversation)
            texts.append(text_prompt)

        #Prepare inputs (depth_images + text)
        depth_batch = self.processor(
            images=depth_images,
            text=texts,
            return_tensors="pt",
            padding=True)
        
        depth_pixel_values = depth_batch["pixel_values"]

        depth_input_ids = depth_batch["input_ids"]

        # Prepare inputs (rgb_images + text)
        rgb_batch  = self.processor(
                images=rgb_images,
                text=texts,
                return_tensors="pt",
                padding=True
            )

        labels = rgb_batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        rgb_batch["labels"] = labels

        
        rgb_input_ids = rgb_batch["input_ids"]
        rgb_pixel_values = rgb_batch["pixel_values"]
        image_sizes = rgb_batch["image_sizes"]
        labels = rgb_batch["labels"]

        question_id = idx


        # Add labels to the batch dictionary
        return {
                "rgb_input_ids": rgb_input_ids,
                "depth_input_ids": depth_input_ids,
                "rgb_pixel_values": rgb_pixel_values,
                "depth_pixel_values": depth_pixel_values,
                "image_sizes": image_sizes,
                "labels": labels,
                "question_id": question_id
            }



