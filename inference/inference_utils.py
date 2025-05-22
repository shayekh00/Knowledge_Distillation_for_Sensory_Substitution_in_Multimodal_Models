import os
import json
import torch
import pandas as pd
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from num2words import num2words
from transformers import LogitsProcessorList, LogitsProcessor
from collections import Counter
import glob
import re
import argparse
from scipy.ndimage import convolve


def initialize_model(MAIN_ROOT_DIR, checkpoint_filename, model_id="llava-hf/llava-onevision-qwen2-0.5b-ov-hf", load_checkpoint=False,
                    model_type="logit_based", phase_param=2,):
    
    """Initialize the processor and model."""

        # Initialize the model
    model_name_student = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    model_name_teacher = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    processor_name = "llava-hf/llava-onevision-qwen2-7b-ov-hf"


    processor = AutoProcessor.from_pretrained( processor_name )

    # checkpoint_file = "llava_onevision_checkpoint_double_trouble_phase3_-epoch=00-val_loss=0.0111.ckpt"
    checkpoint_file = checkpoint_filename

    checkpoint_dir = os.path.join(MAIN_ROOT_DIR, "checkpoints", "kd_checkpoints")
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, checkpoint_file))
    checkpoint_files.sort(key=extract_val_loss)
    checkpoint_path = checkpoint_files[0]

    if model_type == "logit_based":
        from distillation.knowledge_distillation7b_logit_based.OnlineKnowledgeDistillationLLavaOneVision import OnlineKnowledgeDistillationLLavaOneVision
        
    if model_type == "feature_based":
        from distillation.knowledge_distillation7b_feature_based.OnlineKnowledgeDistillationLLavaOneVision import OnlineKnowledgeDistillationLLavaOneVision

    if model_type == "double_trouble":
        from distillation.knowledge_distillation7b_double_trouble.phase1.OnlineKnowledgeDistillationLLavaOneVision import OnlineKnowledgeDistillationLLavaOneVision

    model = OnlineKnowledgeDistillationLLavaOneVision.load_from_checkpoint(
            checkpoint_path,
            model_name_student = model_name_student,
            model_name_teacher = model_name_teacher,
            processor=processor,
            torch_dtype=torch.float16,
            map_location=torch.device('cpu'),
            phase = phase_param
        )
    
    model = model.student_model.to('cuda')
    print("Model loaded from checkpoint at", checkpoint_path)


    
    pad_token_id = (
        processor.tokenizer.eos_token_id
        if processor.tokenizer.pad_token_id is None
        else processor.tokenizer.pad_token_id
    )

    
    model.to("cuda:0")
    
    return processor, model, pad_token_id



def remove_substring_from_path(path, substring="SUNRGBD"):
    """Remove a specific substring from the path."""
    index = path.find(substring)
    if index != -1:
        path = path[:index] + path[index + len(substring):]
    return path

def extract_val_loss(filename):
    # Use regex to find the val_loss value in the filename
    match = re.search(r"val_loss=([\d.]+)\.ckpt", filename)
    if match:
        return float(match.group(1))
    else:
        # If no val_loss is found, return infinity so it's not selected
        return float('inf')


def get_image_paths(img_name_rgb, img_name_depth, root_data_dir):
    """Build and clean the paths for RGB and depth images."""
    directory = os.path.join(root_data_dir, "SUNRGBD")

    # Build full paths
    rgb_image_path = os.path.join(directory, img_name_rgb)
    depth_image_path = os.path.join(directory, img_name_depth)

    # Clean up the paths
    rgb_image_path = remove_substring_from_path(rgb_image_path).replace("\\", "/")
    depth_image_path = remove_substring_from_path(depth_image_path).replace("\\", "/")

    return rgb_image_path, depth_image_path

def convert_numbers_to_words(text):
    """Convert any standalone numbers in the text to words."""
    if text.strip().isdigit():
        return num2words(int(text.strip()))
    words = text.split()
    for i, word in enumerate(words):
        if word.isdigit():
            words[i] = num2words(int(word))
    return " ".join(words)

def convert_depth_image_into_3D(depth_image_path):
    """
    Loads a depth image, applies Prewitt filtering, normalizes the image, 
    and returns a normalized PIL image.
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    # Define Prewitt kernels
    Kx = np.array([[-1,  0,  1],
                   [-1,  0,  1],
                   [-1,  0,  1]], dtype=np.float32)
    
    Ky = np.array([[-1, -1, -1],
                   [ 0,  0,  0],
                   [ 1,  1,  1]], dtype=np.float32)

    def safe_normalize(arr):
        a_min, a_max = arr.min(), arr.max()
        if a_max == a_min:
            a_max = a_min + 1e-6  # Avoid division by zero
        return (arr - a_min) / (a_max - a_min)  # Normalize to [0, 1]

    # 1. Load depth image
    depth_image = Image.open(depth_image_path).convert("I")  # "I" mode is for 32-bit grayscale
    depth_array = np.array(depth_image, dtype=np.float32)

    # 2. Normalize depth to [0, 255]
    depth_norm = (safe_normalize(depth_array) * 255).astype(np.uint8)

    # 3. Apply Prewitt kernels
    Gx = convolve(depth_norm.astype(np.float32), Kx, mode='reflect')
    Gy = convolve(depth_norm.astype(np.float32), Ky, mode='reflect')

    # 4. Compute gradient magnitude and direction
    Gm = np.sqrt(Gx**2 + Gy**2)
    Gtheta = np.arctan2(Gy, Gx)  # range: [-pi, pi]

    # 5. Normalize magnitude and direction to [0, 255]
    Gm_norm = (safe_normalize(Gm) * 255).astype(np.uint8)
    Gtheta_norm = (safe_normalize(Gtheta) * 255).astype(np.uint8)

    # 6. Stack into a 3-channel image
    depth_3channel = np.dstack([depth_norm, Gm_norm, Gtheta_norm]).astype(np.float32) / 255.0  # Normalize to [0,1]

    # 7. Apply ImageNet normalization (Subtract mean, divide by std)
    depth_3channel = (depth_3channel - mean) / std

    # 8. Convert back to 8-bit integer values (0-255 range)
    depth_3channel = np.clip(depth_3channel * 255, 0, 255).astype(np.uint8)

    # 9. Convert back to PIL image
    depth_3channel_pil = Image.fromarray(depth_3channel, mode="RGB")

    return depth_3channel_pil



def get_model_answer(question, image, processor, model, pad_token_id):
    """Get the model's answer to a question given an image."""

    text_prompt = (
        question + " Answer in one word if possible."
    )
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text_prompt},
            ],
        },
    ]
    prompt = processor.apply_chat_template(
        conversation, add_generation_prompt=True
    )
    inputs = processor(
        images=image, text=prompt, return_tensors="pt"
    ).to("cuda:0", torch.float16)



    # Autoregressively complete prompt
    with torch.no_grad():
        output = model.generate(
            # **inputs, max_new_tokens=32,logits_processor=logits_processor, pad_token_id=pad_token_id
            **inputs,
            max_new_tokens=32,
            # logits_processor=logits_processor,
            pad_token_id=pad_token_id,
            repetition_penalty=1.2,  # Penalizes repeated sequences
            no_repeat_ngram_size=2,  # Prevents repeating bigrams
            temperature=0.7  # Makes output more controlled
        )

    model_answer = processor.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    # print("Model Main Answer:", model_answer)

    # Extract the assistant's response
    split_text = model_answer.split("assistant")
    # print(split_text)
    if len(split_text) > 1:
        final_answer = split_text[1].strip().lower()
    else:
        final_answer = model_answer.strip().lower()

    final_answer = convert_numbers_to_words(final_answer)

    return final_answer

def load_environment():
    """Load environment variables."""
    load_dotenv()
    # root_data_dir = os.getenv("root_data_dir")
    ROOT_DATA_DIR = os.getenv("ROOT_DATA_DIR")
    MAIN_ROOT_DATA_DIR = os.getenv("MAIN_ROOT_DATA_DIR")

    return ROOT_DATA_DIR , MAIN_ROOT_DATA_DIR
