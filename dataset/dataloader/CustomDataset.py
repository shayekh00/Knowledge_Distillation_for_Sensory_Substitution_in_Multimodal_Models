import pandas as pd
from transformers import BertTokenizer, BertModel
import os
from PIL import Image
import torchvision.transforms.functional as TF
import random
import torchvision.transforms as transforms

def remove_substring_from_path(path, substring="SUNRGBD"):
    index = path.find(substring)
    if index != -1:
        path = path[:index] + path[index + len(substring):]
    
    # new_path = path.replace(substring, "")

    return path

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, augmentation=True):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.augmentation = augmentation
        
        # Initialize BERT tokenizer and model
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        
        self.dataset_directory = 'dataset/output_path/SUNRGBD/'

        # Tokenization and building vocabulary for answers
        self.answer_to_idx = {}
        self.idx_to_answer = {}
        for answer in self.df['Answers']:
            if answer not in self.answer_to_idx:
                idx = len(self.answer_to_idx)
                self.answer_to_idx[answer] = idx
                self.idx_to_answer[idx] = answer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name_rgb = self.df.iloc[idx, 2]
        img_name_depth = self.df.iloc[idx, 3]
        directory = self.dataset_directory
        rgb_image_path = os.path.join(directory, img_name_rgb) 
        depth_image_path = os.path.join(directory, img_name_depth) 

        rgb_image_path = remove_substring_from_path(rgb_image_path).replace("\\", "/")
        depth_image_path = remove_substring_from_path(depth_image_path).replace("\\", "/")

        rgb_image = Image.open(rgb_image_path)
        depth_image = Image.open(depth_image_path)

        question = self.df.iloc[idx, 0]
        answer = self.df.iloc[idx, 1]

        # # Encode question using BERT tokenizer and model
        # question_encoding = self.tokenizer.encode_plus(question, add_special_tokens=True, max_length=128, truncation=True, padding='max_length', return_tensors='pt')
        # question_embedding = self.bert_model(input_ids=question_encoding['input_ids'], attention_mask=question_encoding['attention_mask'])[0]

        question_encoding = tokenizer.encode_plus(question, add_special_tokens=True, max_length=128, truncation=True, padding='max_length', return_tensors='pt')

        answer_tokens = self.answer_to_idx[answer]

        if self.transform:
            if self.augmentation:
                # Random horizontal flip
                if random.random() > 0.5:
                    rgb_image = TF.hflip(rgb_image)
                    depth_image = TF.hflip(depth_image)

            rgb_image = self.transform(rgb_image)
            depth_image = transforms.ToTensor()(depth_image)
            # depth_image = depth_image.repeat(3, 1, 1)  # Repeat the single channel 3 times
            depth_image = depth_image.float()  # Convert to float
            depth_image = depth_image / 255.0  # Normalize to [0,1]
            depth_image = TF.to_pil_image(depth_image)
            depth_image = self.transform(depth_image)  # Normalize further

            if self.augmentation:
                # Random rotation
                angle = random.randint(-10, 10)
                rgb_image = TF.rotate(rgb_image, angle)
                depth_image = TF.rotate(depth_image, angle)

            
        return rgb_image, depth_image, question_encoding['input_ids'].squeeze(0), question_encoding['attention_mask'].squeeze(0), answer_tokens
        # return rgb_image, depth_image, question_embedding, answer_tokens