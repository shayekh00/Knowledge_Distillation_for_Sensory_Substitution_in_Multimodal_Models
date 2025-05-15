import sys
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoProcessor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset.dataloader.Florence.CustomSUNRGBDDataset import CustomSUNRGBDDataset
import traceback


config_file_path = "models/Florence2/Florence_2_base"
processor = AutoProcessor.from_pretrained(
                    config_file_path, trust_remote_code=True, revision="refs/pr/6"
                )

class CustomSUNRGBDDatasetModule(pl.LightningDataModule):
    def __init__(self, root_data_dir, batch_size, config_file_path ,num_workers, subset_percentage=None):
        super().__init__()
        self.root_data_dir = root_data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.config_file_path = config_file_path
        self.subset_percentage = subset_percentage


    def setup(self, stage=None):
        self.train_dataset = CustomSUNRGBDDataset(
            ROOT_DATA_DIR=self.root_data_dir, csv_file_name="train_dataset.csv",subset_percentage=self.subset_percentage
        )
        self.val_dataset = CustomSUNRGBDDataset(
            ROOT_DATA_DIR=self.root_data_dir, csv_file_name="val_dataset.csv",subset_percentage=self.subset_percentage
        )
        self.test_dataset = CustomSUNRGBDDataset(
            ROOT_DATA_DIR=self.root_data_dir, csv_file_name="test_dataset.csv",subset_percentage=self.subset_percentage
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,  
            pin_memory=True  
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,  
            pin_memory=True  
        )

    @staticmethod
    def collate_fn(batch):
        questions, answers, rgb_images, depth_images, idx= zip(*batch)
        try:
            student_inputs = processor(
                text=list(questions),
                images=list(depth_images),
                return_tensors="pt",
                padding=True,
            )
            teacher_inputs = processor(
                text=list(questions), images=list(rgb_images), return_tensors="pt", padding=True
            )
            
        except Exception as e:
            # Handle the error by printing the traceback and the data causing it
            print(f"Error processing the data: {idx}, {questions}, {answers}, {rgb_images}")
            print(f"Exception: {e}")
            traceback.print_exc()

            


        return student_inputs, teacher_inputs, answers, idx