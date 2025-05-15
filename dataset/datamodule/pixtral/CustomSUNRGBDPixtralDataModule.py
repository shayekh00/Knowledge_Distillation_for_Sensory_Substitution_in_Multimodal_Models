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


class CustomSUNRGBDPixtralDataModule(pl.LightningDataModule):
    def __init__(self, root_data_dir,processor, batch_size, num_workers, subset_percentage=None):
        super().__init__()
        self.root_data_dir = root_data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers  
        self.subset_percentage = subset_percentage
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.processor = processor


        CHAT_TEMPLATE = """
        {%- for message in messages %} 
        {%- if message.role == "user" %} 
            <s>[INST] 
            {%- for item in message.content %} 
            {%- if item.type == "text" %} 
                {{ item.text }} 
            {%- elif item.type == "image" %} 
                \n[IMG] 
            {%- endif %} 
            {%- endfor %} 
            [/INST] 
        {%- elif message.role == "assistant" %} 
            {%- for item in message.content %} 
            {%- if item.type == "text" %} 
                {{ item.text }} 
            {%- endif %} 
            {%- endfor %} 
            </s>
        {%- endif %} 
        {%- endfor %} 
        """

        # Set the chat template for the tokenizer
        self.processor.chat_template = CHAT_TEMPLATE.replace('  ', '')

        self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        

    def setup(self, stage=None):
        # self.processor = self.processor
        
        self.train_dataset = CustomSUNRGBDDatasetOneVision(
            root_data_dir=self.root_data_dir,
            csv_file_name="train_dataset.csv",
            subset_percentage=self.subset_percentage,
        )
        self.val_dataset = CustomSUNRGBDDatasetOneVision(
            root_data_dir=self.root_data_dir,
            csv_file_name="val_dataset.csv",
            subset_percentage=self.subset_percentage,
        )
        self.test_dataset = CustomSUNRGBDDatasetOneVision(
            root_data_dir=self.root_data_dir,
            csv_file_name="test_dataset.csv",
            subset_percentage=self.subset_percentage,
        )


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
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
        #https://github.com/tomstaan/Clarivex-Pixtral-12B/blob/f7c65114dedf4af4859fc680aab4682581c4829a/pixtral_finetune.ipynb#L41
        # Implementation taken from the above link  
        rgb_images = []
        depth_images = []
        idxs = []
        texts = []
        assistant_responses = []


        
        for question, answer, rgb_image_np, depth_image_3channel_array, idx in batch:

            additional_instructions = ".Answer in one word.Don't use the word 'based'."
            final_question = question + additional_instructions



            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": final_question},
                        {"type": "image"},  # Images after the text.
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": answer}
                    ]
                }
            ]

            text = self.processor.apply_chat_template(messages, add_generation_prompt=False)
            texts.append(text.strip())
            # questions.append(question)
            idxs.append(idx)
            rgb_images.append(Image.fromarray(rgb_image_np))
            depth_images.append(Image.fromarray(depth_image_3channel_array))
            assistant_responses.append(answer)

        # additional_instructions = ".Answer in one word.Don't use the word 'based'."
        # text_prompt = questions[0]
        # text_prompts = ["<s>[INST]"+text_prompt+additional_instructions+"\n[IMG][/INST]"]
    
        # Now process them as a batch
        rgb_batch = self.processor(
            images=rgb_images,
            text=texts,
            return_tensors="pt",
            padding=True

        )
        depth_batch = self.processor(
            images=depth_images,
            text=texts,
            return_tensors="pt",
            padding=True
         )
        
        labels = rgb_batch["input_ids"].clone()
        
        # For each example, find assistant tokens and mask everything else
        for i, (input_ids, assistant_response) in enumerate(zip(rgb_batch["input_ids"], assistant_responses)):
            # Tokenize just the assistant response
            assistant_tokens = self.processor.tokenizer(assistant_response, return_tensors="pt")["input_ids"][0]

            # Find where the assistant tokens start in the input sequence
            start_idx = self.find_subsequence(input_ids, assistant_tokens)

            if start_idx is not None:
                # Mask everything except the assistant tokens
                labels[i, :start_idx] = -100  # Ignore everything before the assistant's response
                labels[i, start_idx + len(assistant_tokens):] = -100  # Ignore everything after the assistant's response

        # Assign masked labels back to the batch
        rgb_batch["labels"] = labels
        depth_batch["labels"] = labels

        # # These are now Tensors of shape [batch_size, 3, H, W]
        rgb_pixel_values = rgb_batch["pixel_values"]      # Torch tensor
        depth_pixel_values = depth_batch["pixel_values"]  # Torch tensor

        # rgb_pixel_values = rgb_pixel_values[0] if rgb_pixel_values is not None else None
        # depth_pixel_values = depth_pixel_values[0] if depth_pixel_values is not None else None

        # print(rgb_pixel_values)

        # # Prepare labels
        # labels = rgb_batch["input_ids"].clone()
        # labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "rgb_input_ids": rgb_batch["input_ids"],
            "depth_input_ids": depth_batch["input_ids"],
            "rgb_pixel_values": rgb_pixel_values,
            "depth_pixel_values": depth_pixel_values,
            "labels": labels,
            "question_id": idxs,
        }
    
    def find_subsequence(self, sequence, subsequence):
        """
        Find the start index of a subsequence (assistant tokens) in a sequence (input tokens).
        """
        seq_len = len(sequence)
        sub_len = len(subsequence)

        for i in range(seq_len - sub_len + 1):
            if torch.equal(sequence[i:i + sub_len], subsequence):
                return i
        return None


