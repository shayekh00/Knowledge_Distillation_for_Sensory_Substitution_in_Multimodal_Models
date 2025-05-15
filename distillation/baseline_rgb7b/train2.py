from transformers import AutoProcessor, AutoConfig,AutoModelForCausalLM
import os
import argparse
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from pytorch_lightning import Trainer
# from distillation.onevision_llava.LLavaOneVisionModule import LLaVAOneVisionModule
from dataset.dataloader.OneVision.CustomSUNRGBDDatasetOneVision import CustomSUNRGBDDatasetOneVision
from dataset.datamodule.OneVision.CustomSUNRGBDOneVisionDataModule import CustomSUNRGBDOneVisionDataModule
from llava.model.builder import load_pretrained_model
import pytorch_lightning as pl
from distillation.onevision_llava.LLavaOneVisionModule import LlavaOnevisionModule
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import glob
import re

from dotenv import load_dotenv
import sys
# from callbacks.SaveModelCallback import SaveModelCallback
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../')))




load_dotenv()
root_data_dir = os.getenv("root_data_dir2")
print("root_data_dir:", root_data_dir)
MAIN_ROOT_DATA_DIR = os.path.abspath(root_data_dir)
ROOT_DATA_DIR_DATA = os.path.abspath(root_data_dir + "data/" )
current_dir = os.getcwd()
print("Current directory:", current_dir)


import os
import torch

def extract_val_loss(filename):
    # Use regex to find the val_loss value in the filename
    match = re.search(r"val_loss=([\d.]+)\.ckpt", filename)
    if match:
        return float(match.group(1))
    else:
        # If no val_loss is found, return infinity so it's not selected
        return float('inf')
    
def main():

    parser = argparse.ArgumentParser(description="LLaVA OneVision Training Script")

    # Add arguments for batch_size and max_epochs
    parser.add_argument('--batch_size', type=int, default=1, help="Input batch size for training (default: 1)")
    parser.add_argument('--max_epochs', type=int, default=10, help="Maximum number of epochs to train (default: 10)")
    parser.add_argument('--subset_percentage', type=float, default=1, help="Percentage of the dataset to use (default: 0.01)")
    parser.add_argument('--load_checkpoint', action='store_true', help="Flag to load from checkpoint if set")
    parser.add_argument('--augmentation', action='store_true', help="Enable data augmentation for training")
    parser.add_argument('--accumulate_grad_batches', type=int, default=64, help="Input batch size for training (default: 64)")

    print("Arguments:", parser.parse_args())
    args = parser.parse_args()
    # Initialize the model
    # model_name = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    model_name = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    processor_name = "llava-hf/llava-onevision-qwen2-7b-ov-hf"

    processor = AutoProcessor.from_pretrained(processor_name)

    checkpoint_dir = os.path.join(MAIN_ROOT_DATA_DIR, "checkpoints", "baseline7b_rgb")
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "llava-onevision7b*.ckpt"))

    if args.load_checkpoint:
        
        # checkpoint_filename = "llava-onevision7b-epoch=00-val_loss=0.0095.ckpt"
        # checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

        # Sort the files by validation loss (lowest first)
        checkpoint_files.sort(key=extract_val_loss)        
        # Load the checkpoint with the lowest validation loss
        checkpoint_path = checkpoint_files[0]

        model = LlavaOnevisionModule.load_from_checkpoint(
                checkpoint_path,
                model_name=model_name,
                processor=processor,
                torch_dtype=torch.float16,
            )
        print("Model loaded from checkpoint at", checkpoint_path)
    else:
        print("Model loaded from Pre-Trained")
        model = LlavaOnevisionModule(model_name, processor)

    
    # print(model.model)


    # Initialize the data module
    data_module = CustomSUNRGBDOneVisionDataModule(
        root_data_dir=ROOT_DATA_DIR_DATA,
        processor=processor,
        batch_size=args.batch_size,
        num_workers=4,
        subset_percentage=args.subset_percentage,  # Adjust as needed
        augmentation=args.augmentation  # Enable data augmentation
    )

    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='llava-onevision7b-{epoch:02d}-{val_loss:.5f}',
        save_top_k=1,
        monitor='val_loss'
    )

    # Set up logger
    log_name = f"llava_onevision_7b_batch_size{args.batch_size}_epochs{args.max_epochs}_grad_accum{args.accumulate_grad_batches}_RGB_{'aug' if args.augmentation else 'noaug'}"
    logger = TensorBoardLogger("tb_logs", name= log_name)
    # save_model_callback = SaveModelCallback(save_dir='checkpoints/baseline', filename='llava-onevision-epoch{epoch:02d}.pt')

    # Initialize the Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        # accelerator="gpu" if torch.cuda.is_available() else "cpu",
        accelerator="gpu",
        precision="bf16-true",  
        callbacks=[checkpoint_callback],
        logger=logger,
        check_val_every_n_epoch=1,    
        num_sanity_val_steps=0,         # Skip data sanity check
        fast_dev_run=False,             # Skip training
        accumulate_grad_batches= args.accumulate_grad_batches

    )

    # Start training
    trainer.fit(model, datamodule=data_module)



if __name__ == "__main__":
    main()
