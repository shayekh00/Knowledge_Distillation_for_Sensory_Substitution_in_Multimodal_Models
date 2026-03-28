import os
import json
import torch
import cv2
import numpy as np
from PIL import Image

# Set huggingface endpoints
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "./model_checkpoint/hf_cache"

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from Advanced_IGOS_PP.utils import *
from Advanced_IGOS_PP.methods_helper import *
from Advanced_IGOS_PP.IGOS_pp import *

def main():
    print("Creating dummy image and dataset...")
    # Create a dummy image
    dummy_img_path = "dummy_image.jpg"
    if not os.path.exists(dummy_img_path):
        dummy_img = np.zeros((336, 336, 3), dtype=np.uint8)
        dummy_img[:, :] = (200, 100, 50)
        cv2.imwrite(dummy_img_path, dummy_img)

    text_prompt = "Describe the image in one factual English sentence."
    
    print("Loading model and processor...")
    # Using Qwen2.5-VL-3B since it's already coded in their script
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "katuni4ka/tiny-random-qwen2.5-vl", torch_dtype="auto", device_map="cpu"
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
        
    processor = AutoProcessor.from_pretrained("katuni4ka/tiny-random-qwen2.5-vl")
    tokenizer = processor.tokenizer
    explainer = gen_explanations_qwenvl

    print("Running explainer...")
    image = Image.open(dummy_img_path).convert('RGB')
    
    # We pass it to the explainer
    heatmap, superimposed_img = explainer(model, processor, image, text_prompt, tokenizer)
    
    # Output paths
    output_npy = "heatmap.npy"
    output_vis = "visualization.jpg"
    
    np.save(output_npy, np.array(heatmap))
    cv2.imwrite(output_vis, superimposed_img)
    
    print(f"Success! Output saved to {output_vis} and {output_npy}")

if __name__ == "__main__":
    main()
