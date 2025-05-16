import os
import json
import torch
import pandas as pd
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from num2words import num2words
from transformers import LogitsProcessorList, LogitsProcessor
from collections import Counter
import glob
import re
import argparse
from scipy.ndimage import convolve


import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
print("sys.path:", sys.path)
current_dir = os.getcwd()
print("Current directory:", current_dir)


import spacy
'''
Usage: 
python evaluation/onevisionv3/sunrgbd/evaluate_onevision_kd_new.py --model_id llava-hf/llava-onevision-qwen2-0.5b-ov-hf --gts_type val --load_checkpoint
python evaluation/onevisionv3/sunrgbd/evaluate_onevision_kd_new.py --model_id llava-hf/llava-onevision-qwen2-7b-ov-hf --gts_type val --load_checkpoint
'''



def initialize_model(MAIN_ROOT_DIR, model_id="llava-hf/llava-onevision-qwen2-0.5b-ov-hf", load_checkpoint=False, model_type="logit_based", phase_param=2):
    """Initialize the processor and model."""

        # Initialize the model
    model_name_student = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    model_name_teacher = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    processor_name = "llava-hf/llava-onevision-qwen2-7b-ov-hf"


    processor = AutoProcessor.from_pretrained( processor_name )

    checkpoint_file = "llava_onevision_checkpoint_double_trouble_phase3_-epoch=00-val_loss=0.0111.ckpt"

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
   
def load_environment():
    """Load environment variables."""
    load_dotenv()
    # root_data_dir = os.getenv("root_data_dir")
    root_data_dir = XXXX
    print("Root data directory:", root_data_dir)
    if root_data_dir is None:
        raise ValueError("Environment variable 'root_data_dir' not set.")
    
    
    MAIN_ROOT_DATA_DIR = os.path.abspath(root_data_dir)
    ROOT_DATA_DIR_DATA = os.path.abspath(root_data_dir + "data/" )
    return ROOT_DATA_DIR_DATA , MAIN_ROOT_DATA_DIR

def extract_val_loss(filename):
    # Use regex to find the val_loss value in the filename
    match = re.search(r"val_loss=([\d.]+)\.ckpt", filename)
    if match:
        return float(match.group(1))
    else:
        # If no val_loss is found, return infinity so it's not selected
        return float('inf')
    

def remove_substring_from_path(path, substring="SUNRGBD"):
    """Remove a specific substring from the path."""
    index = path.find(substring)
    if index != -1:
        path = path[:index] + path[index + len(substring):]
    return path

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

class RestrictedLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = set(allowed_token_ids)

    def __call__(self, input_ids, scores):
        mask = torch.full(scores.shape, float('-inf')).to(scores.device)
        mask[:, list(self.allowed_token_ids)] = 0  # Set allowed token positions to 0 (unmasked)
        return scores + mask
    
def get_logit_processor(answers_df ,processor ):
    
    answer_tokens = set(answers_df['Unique Answers'].str.split().sum())  # Collect unique tokens in "Answers" column

    tokenizer = processor.tokenizer
    allowed_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in answer_tokens if token in tokenizer.get_vocab()]
    logits_processor = LogitsProcessorList([RestrictedLogitsProcessor(allowed_token_ids)])

    return logits_processor

def get_model_answer(question, image, processor, model, pad_token_id,logits_processor):
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

def convert_depth_image(depth_image_path):
    """Convert a single-channel depth image to a normalized three-channel image."""
    depth_image = Image.open(depth_image_path)
    depth_array = np.array(depth_image)

    depth_min = depth_array.min()
    depth_max = depth_array.max()

    # Avoid division by zero
    if depth_max - depth_min == 0:
        depth_normalized = np.zeros_like(depth_array, dtype=np.uint8)
    else:
        depth_normalized = (
            255 * (depth_array - depth_min) / (depth_max - depth_min)
        ).astype(np.uint8)
    depth_3channel = np.stack([depth_normalized] * 3, axis=-1)
    depth_image_pil = Image.fromarray(depth_3channel)

    return depth_image_pil



mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


def convert_depth_image_into_3D(depth_image_path):
    """
    Loads a depth image, applies Prewitt filtering, normalizes the image, 
    and returns a normalized PIL image.
    """
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

def main():
    '''
    
    Usage: 
    python evaluation/onevisionv3/sunrgbd/evaluate_onevision.py --model_id llava-hf/llava-onevision-qwen2-0.5b-ov-hf --gts_type val --load_checkpoint
    python evaluation/onevisionv3/sunrgbd/evaluate_onevision.py --model_id llava-hf/llava-onevision-qwen2-7b-ov-hf --gts_type val --load_checkpoint
    '''

    parser = argparse.ArgumentParser(description="Script for loading and initializing a model.")
    parser.add_argument("--model_id", type=str, required=True, help="The ID of the model to use.")
    parser.add_argument("--gts_type", type=str, choices=["val", "test"], required=True, help="The type of dataset to use (val or test).")
    parser.add_argument("--load_checkpoint", action="store_true", help="Whether to load the model from a checkpoint.")

    # Parse the arguments
    args = parser.parse_args()

    # Assign arguments to variables
    model_id = args.model_id
    gts_type = args.gts_type
    load_checkpoint = args.load_checkpoint

    # if "0.5b" in model_id:
    #     model_used = "0.5b"
    # else:
    #     model_used = "7b"


    # kd_model_type = "logit_based"
    # kd_model_type = "feature_based"
    kd_model_type = "double_trouble"

    if kd_model_type == "double_trouble":
        phase_no = 3
        phase = "phase"+str(phase_no)
        phase_param = phase_no
    else:
        phase = "_"



    # model_id = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    # gts_type = "val"  # Change to "test" for test dataset
    # load_checkpoint = True

    # Load environment variables and initialize the model
    ROOT_DATA_DIR, MAIN_ROOT_DIR = load_environment()

    processor, model, pad_token_id = initialize_model(MAIN_ROOT_DIR, model_id="llava-hf/llava-onevision-qwen2-0.5b-ov-hf", load_checkpoint=False, model_type=kd_model_type, phase_param=phase_param)
    # processor, model, pad_token_id = initialize_model(MAIN_ROOT_DIR, model_id, load_checkpoint)
    # #getting the model from the module
    # model = model.model

    
    unique_tokens_path = os.path.join(ROOT_DATA_DIR, "SUNRGBD/", "unique_tokens_new5.csv")

    answers_df = pd.read_csv(unique_tokens_path)
    # answers_df = pd.read_csv("dataset/SUNRGBD_Dataset/unique_tokens_new4.csv")

    logits_processor = get_logit_processor(answers_df, processor)
    # Paths to CSV files

    csv_file_path_val = os.path.join(ROOT_DATA_DIR, "SUNRGBD/csv_data", "val_dataset.csv")
    csv_file_path_test = os.path.join(ROOT_DATA_DIR, "SUNRGBD/csv_data", "test_dataset.csv")

    # print("Train DF length:", len(pd.read_csv(csv_file_path_train)))
    # Read the CSV files
    df_val = pd.read_csv(csv_file_path_val)
    df_test = pd.read_csv(csv_file_path_test)



    if gts_type == "val":
        df = df_val
    else:
        df = df_test

    # ds_len = min(len(df), 100000)  # Limit to n samples for testing
    ds_len = len(df)  # Limit to n samples for testing

    # Directory to save images with incorrect answers
    output_dir = os.path.join(
        "dataset", "SUNRGBD_Dataset", "sample_check", "processed_images"
    )
    os.makedirs(output_dir, exist_ok=True)

        # Load the spaCy language model
    nlp = spacy.load("en_core_web_md")


    results = []
    correct = 0
    incorrect = 0
    data_count = 0
    predictions = []
    references = []
    incorrect_df = pd.DataFrame(columns=['Question_Id', 'Questions', 'Question_Type', 'Answers',"Model_Answer"])
    correct_df = pd.DataFrame(columns=['Question_Id', 'Questions', 'Question_Type', 'Answers',"Model_Answer"])
    results_df = pd.DataFrame(columns=['Question_Id', 'Questions', 'Question_Type', 'Answers',"Model_Answer"])
    data_list = []
    pixel_data_type = "depth"
    
    print("Image used :", pixel_data_type)

    # bertscore = BERTScore(lang="en")

    #Loop through the dataset
    progress_bar = tqdm(range(ds_len))
    for i in progress_bar:
        # try:
        row = df.iloc[i]
        question_id = int(row["Question_Id"])
        question = row["Questions"]
        question_type = df.iloc[i]['Question_Type']
        answers = row["Answers"].strip().lower()

        rgb_image_path, depth_image_path = get_image_paths(
            row["Image_Path"], row["Depth_Path"], ROOT_DATA_DIR
        )

        rgb_image = Image.open(rgb_image_path)
        rgb_image_np = np.array(rgb_image)

        depth_image_np = convert_depth_image_into_3D(depth_image_path )
        # Optionally process depth image if needed
        # depth_image = convert_depth_image(depth_image_path)
        if pixel_data_type == "depth":
            image_used = depth_image_np 
        else:
            image_used = rgb_image_np

        
        model_final_answer = get_model_answer(
            question, image_used, processor, model, pad_token_id,logits_processor
        )
        

        data_list.append({
            'Question_Id': question_id,
            'Questions': question,
            'Question_Type': question_type,
            'Answers': answers,
            'Model_Answer': model_final_answer
        })

        predictions.append(model_final_answer)
        references.append(answers)


    #Write the results to a CSV file
    results_df = pd.DataFrame(data_list)
    
    path_to_save = os.path.join( "dataset/predictions/", "results_kd_modeltype:"+pixel_data_type+"_"+gts_type+"_"+kd_model_type+phase+".csv")
    
    results_df.to_csv(path_to_save, index=False)  
    print("Results saved to:", path_to_save)


if __name__ == "__main__":
    main()
