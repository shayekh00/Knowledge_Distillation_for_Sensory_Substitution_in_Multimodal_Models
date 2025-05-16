from transformers import AutoProcessor, AutoConfig,AutoModelForCausalLM
import os
import sys
import argparse
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from pytorch_lightning import Trainer
# from distillation.onevision_llava.LLavaOneVisionModule import LLaVAOneVisionModule
from dataset.dataloader.OneVision.CustomSUNRGBDDatasetOneVision import CustomSUNRGBDDatasetOneVision
from dataset.datamodule.OneVision.CustomSUNRGBDOneVisionDataModule import CustomSUNRGBDOneVisionDataModule
from llava.model.builder import load_pretrained_model
import pytorch_lightning as pl
from distillation.knowledge_distillation7b_double_trouble.phase1.OnlineKnowledgeDistillationLLavaOneVision import OnlineKnowledgeDistillationLLavaOneVision
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import sys
import re
import glob
from pytorch_lightning.strategies import DDPStrategy ,FSDPStrategy
from transformers import LlavaOnevisionForConditionalGeneration
import torch.nn as nn
from pytorch_lightning.callbacks import TQDMProgressBar
from dotenv import load_dotenv
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../../')))

load_dotenv()

ROOT_DATA_DIR = os.getenv("ROOT_DATA_DIR")
MAIN_ROOT_DATA_DIR = os.getenv("MAIN_ROOT_DATA_DIR")
current_dir = os.getcwd()


print("Current directory:", current_dir)

hf_token = os.getenv("hf_token")
# print("hf_token:", hf_token)    
from huggingface_hub import login

login(hf_token)

torch.cuda.empty_cache()
torch.cuda.ipc_collect()

def extract_val_loss(filename):
    # Use regex to find the val_loss value in the filename
    match = re.search(r"val_loss=([\d.]+)\.ckpt", filename)
    if match:
        return float(match.group(1))
    else:
        # If no val_loss is found, return infinity so it's not selected
        return float('inf')
    
def main():
    '''
    Example command: 
    python distillation/knowledge_distillation7b_double_trouble/phase2/train_online_kd.py --batch_size 1 --max_epochs 10 --subset_percentage 1 --load_checkpoint
    
    '''

    parser = argparse.ArgumentParser(description="LLaVA OneVision Training Script")

    # Add arguments for batch_size and max_epochs
    parser.add_argument('--batch_size', type=int, default=1, help="Input batch size for training (default: 1)")
    parser.add_argument('--max_epochs', type=int, default=10, help="Maximum number of epochs to train (default: 10)")
    parser.add_argument('--subset_percentage', type=float, default=1, help="Percentage of the dataset to use (default: 0.01)")
    parser.add_argument('--load_checkpoint', action='store_true', help="Flag to load from checkpoint if set")
    parser.add_argument('--augmentation', action='store_true', help="Enable data augmentation for training")
    parser.add_argument('--accumulate_grad_batches', type=int, default=64, help="Input batch size for training (default: 64)")


    args = parser.parse_args()

    # Initialize the model
    model_name_student = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    model_name_teacher = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    processor_name = "llava-hf/llava-onevision-qwen2-7b-ov-hf"

    checkpoint_dir = os.path.join(MAIN_ROOT_DATA_DIR, "checkpoints", "kd_checkpoints")
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "llava_onevision_checkpoint_double_trouble_*.ckpt"))
    tensorboard_logs_dir = os.path.join(MAIN_ROOT_DATA_DIR, "tensorboard_logs", "kd_logs")

    processor = AutoProcessor.from_pretrained(processor_name)

    if args.load_checkpoint:
        
        checkpoint_filename = "llava_onevision_checkpoint_double_trouble_-epoch=01-val_loss=0.5270.ckpt"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

        # Sort the files by validation loss (lowest first)
        checkpoint_files.sort(key=extract_val_loss)        
        # Load the checkpoint with the lowest validation loss
        checkpoint_path = checkpoint_files[0]

        print("Model Loading from checkpoint ....", checkpoint_path)
        model = OnlineKnowledgeDistillationLLavaOneVision.load_from_checkpoint(
                checkpoint_path,
                model_name_student = model_name_student,
                model_name_teacher = model_name_teacher,
                processor=processor,
                torch_dtype=torch.float16,
                map_location=torch.device('cpu'),
                phase=2
            )
        model.freeze_student_vision_layers()
        # model = model.to('cuda')
        print("Model loaded from checkpoint at", checkpoint_path)
    else:
        print("Model loaded from Pre-Trained")
        model = OnlineKnowledgeDistillationLLavaOneVision(model_name_student, model_name_teacher, processor, phase=2)
        model.freeze_student_vision_layers()




    # Initialize the data module
    data_module = CustomSUNRGBDOneVisionDataModule(
        root_data_dir=ROOT_DATA_DIR,
        processor=processor,
        batch_size=args.batch_size,
        num_workers=4,
        subset_percentage=args.subset_percentage  # Adjust as needed
    )

    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='llava_onevision_checkpoint_double_trouble_phase2_-{epoch:02d}-{val_loss:.4f}',
        save_top_k=1,
        monitor='val_loss',
        mode='min'  
    )

    # Set up logger
    log_name = f"kd_double_trouble_phase2_batch{args.batch_size}_epochs{args.max_epochs}_grad_accum{args.accumulate_grad_batches}_RGB_{'aug' if args.augmentation else 'noaug'}"
    logger = TensorBoardLogger(tensorboard_logs_dir, name=log_name)

    # Initialize the Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        # accelerator="gpu" if torch.cuda.is_available() else "cpu",
        accelerator="gpu",
        # precision="bf16-true",
        precision="16",
        # strategy=DDPStrategy(find_unused_parameters=False),  
        callbacks=[checkpoint_callback,TQDMProgressBar(refresh_rate=1)],
        logger=logger,
        check_val_every_n_epoch=1,    # Skip validation
        num_sanity_val_steps=0,         # Skip data sanity check
        fast_dev_run=False,               # Skip training
        devices=1,
        accumulate_grad_batches= args.accumulate_grad_batches 
        # strategy=FSDPStrategy(
        # # auto_wrap_policy=transformer_auto_wrap_policy(
        # #     transformer_layer_cls={nn.TransformerEncoderLayer, nn.TransformerDecoderLayer}  # Specify transformer layer classes
        # # ),
        # activation_checkpointing_policy={nn.TransformerEncoderLayer, nn.TransformerDecoderLayer},
        # sharding_strategy="FULL_SHARD",
        # limit_all_gathers=True,
        # cpu_offload=True)
        
    
    )

    # Start training
    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    main()
