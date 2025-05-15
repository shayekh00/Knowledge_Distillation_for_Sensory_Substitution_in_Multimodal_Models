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
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000


torch.cuda.empty_cache()
torch.cuda.ipc_collect()

from OnlineKnowledgeDistillationLLavaOneVision import OnlineKnowledgeDistillationLLavaOneVision
import spacy

def are_texts_similar(nlp ,model_final_answer, answers, similarity_threshold=0.5):
    """
    Compares two texts semantically and returns True if they are similar.

    Parameters:
        model_final_answer (str): The first text to compare, ensured to be a string.
        answers (list): The second text to compare, provided as a list of strings.
        similarity_threshold (float): The similarity threshold to consider texts similar. Default is 0.7.

    Returns:
        bool: True if texts are similar, False otherwise.
    """
    # Ensure model_final_answer is a string
    if not isinstance(model_final_answer, str):
        raise ValueError("model_final_answer must be a string")

    # Ensure answers is a list and join it into a single string
    if not isinstance(answers, list):
        raise ValueError("answers must be a list of strings")
    answers = " ".join(answers)

    # Strip whitespace from model_final_answer
    model_final_answer = model_final_answer.strip()

    # # Load the spaCy language model
    # nlp = spacy.load("en_core_web_md")

    # Process the texts
    model_final_answer_doc1 = nlp(model_final_answer)
    answers_doc2 = nlp(answers)

    # Compute similarity
    similarity = model_final_answer_doc1.similarity(answers_doc2)

    # Return whether the similarity is above the threshold
    # print("Similarity:", similarity >= similarity_threshold)
    return similarity >= similarity_threshold

# # Example usage
# if __name__ == "__main__":
#     text1 = "bicycle"
#     text2 = "bike"
#     result = are_texts_similar(text1, text2)
#     print(f"Are the texts '{text1}' and '{text2}' similar? {result}")

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
    
def initialize_model(MAIN_ROOT_DIR, model_id="llava-hf/llava-onevision-qwen2-0.5b-ov-hf", load_checkpoint=False):
    """Initialize the processor and model."""

        # Initialize the model
    model_name_student = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    model_name_teacher = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    processor_name = "llava-hf/llava-onevision-qwen2-7b-ov-hf"


    processor = AutoProcessor.from_pretrained( processor_name )
    
    if load_checkpoint:
        #Check which model to load
        if "0.5b" in model_id:
            print("Loading 0.5b model.............")
            model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
            checkpoint_file = "llava_onevision_checkpoint_-epoch=03-val_loss=13.92.ckpt"
            # checkpoint_file = "llava-onevision0.5b*.ckpt"
        else:
            print("Loading 7b model.............")
            model_id = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
            # checkpoint_file = "llava-onevision7b*.ckpt"
            checkpoint_file = "llava-onevision7b-epoch=01-val_loss=0.0082.ckpt"


        checkpoint_dir = os.path.join(MAIN_ROOT_DIR, "checkpoints", "kd_checkpoints")
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, checkpoint_file))
        # Sort the files by validation loss (lowest first)
        checkpoint_files.sort(key=extract_val_loss)
        # print("Checkpoint files:", checkpoint_files)        
        # Load the checkpoint with the lowest validation loss
        checkpoint_path = checkpoint_files[0]



        model = OnlineKnowledgeDistillationLLavaOneVision.load_from_checkpoint(
                checkpoint_path,
                model_name_student = model_name_student,
                model_name_teacher = model_name_teacher,
                processor=processor,
                torch_dtype=torch.float16,
                map_location=torch.device('cpu')
            )
        model = model.student_model.to('cuda')
        print("Model loaded from checkpoint at", checkpoint_path)


    # else:
    #     print("Model loaded from Pre-Trained", model_id)
    #     model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    #             model_id,
    #             torch_dtype=torch.float16,
    #             low_cpu_mem_usage=True,
    #             )



    
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
            **inputs, max_new_tokens=1,logits_processor=logits_processor, pad_token_id=pad_token_id
        )

    model_answer = processor.decode(output[0], skip_special_tokens=True)

    # Extract the assistant's response
    split_text = model_answer.split("assistant")
    # print(split_text)
    if len(split_text) > 1:
        final_answer = split_text[1].strip().lower()
    else:
        final_answer = model_answer.strip().lower()

    final_answer = convert_numbers_to_words(final_answer)

    return final_answer


# def convert_depth_image_into_3D(depth_image_path):
#     """
#     Loads a depth image, applies Prewitt filtering to compute gradient
#     magnitude and direction, then returns a 3-channel PIL image:
#     (1) normalized depth, (2) gradient magnitude, (3) gradient direction
#     """
#     # Define Prewitt kernels
#     Kx = np.array([[-1,  0,  1],
#                 [-1,  0,  1],
#                 [-1,  0,  1]], dtype=np.float32)
#     Ky = np.array([[-1, -1, -1],
#                 [ 0,  0,  0],
#                 [ 1,  1,  1]], dtype=np.float32)

#     # Helper function to safely normalize an array to [0, 255]
#     def safe_normalize(arr):
#         a_min, a_max = arr.min(), arr.max()
#         if a_max == a_min:
#             a_max = a_min + 1e-6
#         return 255.0 * (arr - a_min) / (a_max - a_min)

#     # 1. Load depth image and convert to float
#     depth_image = Image.open(depth_image_path).convert("I")  # "I" for 32-bit grayscale
#     depth_array = np.array(depth_image, dtype=np.float32)

#     # 2. Normalize depth to [0, 255]
#     depth_min = depth_array.min()
#     depth_max = depth_array.max()
#     if depth_max == depth_min:
#         depth_max = depth_min + 1e-6
#     depth_norm = (255.0 * (depth_array - depth_min) / (depth_max - depth_min)).astype(np.uint8)

#     # 3. Apply Prewitt kernels
#     Gx = convolve(depth_norm.astype(np.float32), Kx, mode='reflect')
#     Gy = convolve(depth_norm.astype(np.float32), Ky, mode='reflect')

#     # 4. Compute gradient magnitude and direction
#     Gm = np.sqrt(Gx**2 + Gy**2)
#     Gtheta = np.arctan2(Gy, Gx)  # range: [-pi, pi]

#     # 5. Normalize magnitude and direction to [0, 255]
#     Gm_norm = safe_normalize(Gm).astype(np.uint8)
#     Gtheta_norm = safe_normalize(Gtheta).astype(np.uint8)

#     # 6. Stack into a 3-channel image
#     depth_3channel = np.dstack([depth_norm, Gm_norm, Gtheta_norm])
#     depth_3channel_pil = Image.fromarray(depth_3channel, mode="RGB")

#     return depth_3channel_pil

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

def update_counters(nlp, model_final_answer, row_data, correct,answers, incorrect ,incorrect_df, data_count, question_type, correct_df):
    # print("Model Final Answer1:", model_final_answer, "Answers", answers)
    model_final_answer = model_final_answer.strip()
    
    answers = answers.split()

    if question_type == "Color Identification":
        
        if (model_final_answer in answers) or ((model_final_answer in ['gray', 'grey','white','silver']) and (answers[0] in ['gray', 'grey','white','silver'])):
            correct += 1
            correct_df.loc[len(correct_df)] = row_data
        else:
            incorrect += 1
            incorrect_df.loc[len(incorrect_df)] = row_data

    elif question_type == "Object Identification":
        if are_texts_similar(nlp ,model_final_answer, answers):
            correct += 1
            correct_df.loc[len(correct_df)] = row_data

        else:
            incorrect += 1
            incorrect_df.loc[len(incorrect_df)] = row_data

    else:
        #For Other Question Types
        if ((model_final_answer !="no") and (len(model_final_answer) == 1 or len(model_final_answer) == 2)) or (model_final_answer not in answers):
                incorrect_df.loc[len(incorrect_df)] = row_data
                incorrect +=1
                # print("Model Final Answer2:", model_final_answer, "Answers", answers)
                # print((model_final_answer not in answers))

        # elif are_texts_similar(nlp ,model_final_answer, answers):
        #     correct += 1
        # elif (model_final_answer not in answers):
        #     incorrect += 1
        #     incorrect_df.loc[len(incorrect_df)] = row_data
        else:
            correct_df.loc[len(correct_df)] = row_data
            correct += 1

    data_count += 1

    return correct, incorrect, incorrect_df , data_count, correct_df




def main():
    '''
    Usage: 
    python evaluation/onevisionv2/sunrgbd/evaluate_onevision.py --model_id llava-hf/llava-onevision-qwen2-0.5b-ov-hf --gts_type val --load_checkpoint
    python evaluation/onevisionv2/sunrgbd/evaluate_onevision.py --model_id llava-hf/llava-onevision-qwen2-7b-ov-hf --gts_type val --load_checkpoint
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

    if "0.5b" in model_id:
        model_used = "0.5b"
    else:
        model_used = "7b"

    # model_id = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    # gts_type = "val"  # Change to "test" for test dataset
    # load_checkpoint = True

    # Load environment variables and initialize the model
    ROOT_DATA_DIR, MAIN_ROOT_DIR = load_environment()


    processor, model, pad_token_id = initialize_model(MAIN_ROOT_DIR, model_id, load_checkpoint)
    # #getting the model from the module
    # model = model.model

    # load_dotenv()
    # root_data_dir = os.getenv("root_data_dir2")
    # ROOT_DATA_DIR = os.path.abspath(root_data_dir)
    
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
    incorrect_df = pd.DataFrame(columns=['Question_Id', 'Questions', 'Question_Type', 'Answers',"Model_Answer"])
    correct_df = pd.DataFrame(columns=['Question_Id', 'Questions', 'Question_Type', 'Answers',"Model_Answer"])

    
    #Loop through the dataset
    progress_bar = tqdm(range(ds_len), desc="Processing Results: Correct=0, Incorrect=0, Accuracy=0")
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

        depth_image_np = convert_depth_image_into_3D(depth_image_path)

        # Optionally process depth image if needed
        # depth_image = convert_depth_image(depth_image_path)
        
        
        model_final_answer = get_model_answer(
            question, depth_image_np, processor, model, pad_token_id,logits_processor
        )


        row_data = {
                'Question_Id': question_id,
                'Questions': question,
                'Question_Type': question_type,
                'Answers': answers,
                'Model_Answer':model_final_answer
            }    
        
        correct, incorrect, incorrect_df , data_count ,correct_df= update_counters(nlp ,model_final_answer, row_data, correct,answers, incorrect ,incorrect_df ,data_count, question_type, correct_df)
        # print(f"Data count: {data_count}, Correct count: {correct}, Incorrect count: {incorrect}")
        progress_bar.set_description(f"Processing Results: Correct={correct}, Incorrect={incorrect}, Accuracy={correct/data_count:.3f}")

        results.append({"question_id": question_id, "answer": model_final_answer})
        # except Exception as e:
        #     print(f"Error processing question {question_id}: {e}")
        #     continue
    
    incorrect_path = "dataset/SUNRGBD_Dataset/sample_check/incorrect_predictions/"+'incorrect_predictions_onevision' + model_used + '.csv'
    correct_path = "dataset/SUNRGBD_Dataset/sample_check/correct_predictions/"+'correct_predictions_onevision' + model_used + '.csv'
    incorrect_df.to_csv(incorrect_path, index=False)
    correct_df.to_csv(correct_path, index=False)
    print("Correct predictions saved to: ", correct_path)
    print("Incorrect predictions saved to: ", incorrect_path)
    # Counting the occurrences of each Question_Type
    category_counts = Counter(incorrect_df['Question_Type'])
    sorted_categories = category_counts.most_common()

    # Printing the results
    print("Incorrectly classified question types from highest to lowest frequency:")
    for question_type, count in sorted_categories:
        print(f"{question_type}: {count}")

    print(f"Accuracy: {correct/ds_len}")


if __name__ == "__main__":
    main()
