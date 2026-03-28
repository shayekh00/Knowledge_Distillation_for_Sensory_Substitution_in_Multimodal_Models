import os
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
import numpy as np
import cv2
from PIL import Image

os.environ["HF_HOME"] = "./model_checkpoint/hf_cache"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["TMP"] = "./model_checkpoint/tmp"
os.environ["TMPDIR"] = "./model_checkpoint/tmp"

def main():
    print("Downloading custom weights from huggingface...")
    ckpt_path = hf_hub_download(repo_id='shayekh00/llava-onevision-phase2', filename='llava_onevision_phase2_checkpoint.ckpt')
    
    print("Loading base model: llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
    model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    processor_id = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    
    processor = AutoProcessor.from_pretrained(processor_id)
    # Use load in float16 to keep memory footprint small
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, device_map="cpu")
    
    print("Loading custom weights into base model...")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    
    # Strip common lighting/training prefixes if they exist
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            new_state[k[6:]] = v
        else:
            new_state[k] = v
            
    res = model.load_state_dict(new_state, strict=False)
    print("Model mapped! Missing keys:", len(res.missing_keys), ", Unexpected keys:", len(res.unexpected_keys))
    
    # Create test image
    img = np.zeros((336, 336, 3), dtype=np.uint8)
    img[:, :] = (100, 100, 200)
    cv2.imwrite("test_input.jpg", img)
    image = Image.open("test_input.jpg").convert('RGB')
    
    print("Evaluating test prompt...")
    text_prompt = "Describe the image."
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text_prompt},
                {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device, model.dtype)
    
    # Fix types for input_ids/attention_mask to be long
    inputs["input_ids"] = inputs["input_ids"].to(torch.long)
    inputs["attention_mask"] = inputs["attention_mask"].to(torch.long)
    if "image_sizes" in inputs:
        inputs["image_sizes"] = inputs["image_sizes"].to(torch.long)
    
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=20)
    
    text = processor.decode(output[0], skip_special_tokens=True)
    print("Model Output:", text)
    print("Integration successful!")

if __name__ == "__main__":
    main()
