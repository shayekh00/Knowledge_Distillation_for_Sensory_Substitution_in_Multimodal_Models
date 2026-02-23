import os
import sys
import base64
import argparse
import pandas as pd
from io import BytesIO
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset
from openai import OpenAI
from dotenv import load_dotenv

# Add inference map so we can import utilities
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SYS_PATH = os.path.join(PROJECT_ROOT, 'inference')
sys.path.append(SYS_PATH)

from inference_utils import (
    convert_depth_image_into_3D,
    get_image_paths
)


def pil_to_base64(img, format="PNG"):
    buffered = BytesIO()
    img.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def get_1d_depth_base64(path):
    # Try opening and converting right to an 8-bit encoded PNG, or directly passing the raw file bytes
    # OpenAI accepts standard formats like PNG/JPEG natively. Raw .png bytes are usually best.
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def query_openai(client, base64_image, question):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": f"{question} Answer in one word if possible."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            # Using image/png broadly; if the original was JPEG, the mimetype might be slightly wrong
                            # but OpenAI API is usually lenient.
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=32
    )
    return response.choices[0].message.content.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_rows", type=int, default=None, help="Max rows to process per split (useful for testing)")
    args = parser.parse_args()
    
    # Load .env variables (fetches OPENAI_API_KEY)
    load_dotenv(os.path.join(PROJECT_ROOT, '.env'))
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY is missing from your .env file!")
        return
        
    client = OpenAI(api_key=api_key)
    
    dataset_repo = "shayekh00/VQA_SUNRGBD_v2"
    print(f"Loading dataset: {dataset_repo}")
    dataset = load_dataset(dataset_repo)
    
    # Using the parent of SUNRGBD as root_data_dir for get_image_paths compatibility
    root_data_dir = os.path.join(PROJECT_ROOT, "dataset")
    splits = ["train", "validation", "test"]
    
    data_list_1d = []
    data_list_3d = []
    
    skip_1d = False  # Flag to cancel 1D requests if they fail
    
    for split in splits:
        if split not in dataset:
            continue
        
        print(f"\n========== Processing split: {split.upper()} ==========")
        split_data = dataset[split]
        
        for i, row in tqdm(enumerate(split_data), total=args.max_rows if args.max_rows else len(split_data)):
            if args.max_rows and i >= args.max_rows:
                break
                
            question_id = row.get("Question_Id")
            question = row.get("Questions")
            question_type = row.get("Question_Type")
            answers = row.get("Answers")
            depth_path_suffix = row.get("Depth_Path")
            image_path_suffix = row.get("Image_Path")
            
            if not depth_path_suffix:
                continue
            
            # Use get_image_paths logic to resolve correct paths
            _, absolute_depth_path = get_image_paths(
                image_path_suffix if image_path_suffix else "dummy.jpg", 
                depth_path_suffix, 
                root_data_dir
            )
            
            if not os.path.exists(absolute_depth_path):
                print(f"[Warning] Depth file missing: {absolute_depth_path} for ID {question_id}")
                continue
            
            # --- 1. 1D Depth Processing ---
            if not skip_1d:
                try:
                    b64_1d = get_1d_depth_base64(absolute_depth_path)
                    model_answer_1d = query_openai(client, b64_1d, question)
                    
                    data_list_1d.append({
                        'Question_Id': question_id,
                        'Questions': question,
                        'Question_Type': question_type,
                        'Answers': answers,
                        'Model_Answer': model_answer_1d
                    })
                except Exception as e:
                    print(f"\n[OpenAI API Error] Failed on 1D depth input for ID {question_id}.")
                    print(f"Error details: {str(e)}")
                    print("--> Aborting 1D depth and continuing exclusively with 3D depth...")
                    skip_1d = True
            
            # --- 2. 3D Depth Processing ---
            try:
                # Use the provided function to map depth image to a 3-channel 3D image
                img_3d = convert_depth_image_into_3D(absolute_depth_path)
                
                # Convert PIL image to base64
                b64_3d = pil_to_base64(img_3d, format="PNG")
                
                # Run through OpenAI
                model_answer_3d = query_openai(client, b64_3d, question)
                
                data_list_3d.append({
                    'Question_Id': question_id,
                    'Questions': question,
                    'Question_Type': question_type,
                    'Answers': answers,
                    'Model_Answer': model_answer_3d
                })
            except Exception as e:
                print(f"\n[Error] Failed on 3D depth processing for ID {question_id}. Error: {e}")

    # --- Saving the results ---
    print("\n========== Saving CSVs ==========")
    output_dir = os.path.join(BASE_DIR, "openai_results")
    os.makedirs(output_dir, exist_ok=True)
    
    if not skip_1d and len(data_list_1d) > 0:
        df_1d = pd.DataFrame(data_list_1d)
        print("\nResults DataFrame (1D Depth):")
        print(df_1d.head())
        out_1d = os.path.join(output_dir, "depth_1d_openai_results.csv")
        df_1d.to_csv(out_1d, index=False)
        print(f"Saved 1D results to: {out_1d}")
    elif skip_1d:
        print("\n1D Depth evaluation was aborted due to API errors.")
    
    if len(data_list_3d) > 0:
        df_3d = pd.DataFrame(data_list_3d)
        print("\nResults DataFrame (3D Depth):")
        print(df_3d.head())
        out_3d = os.path.join(output_dir, "depth_3d_openai_results.csv")
        df_3d.to_csv(out_3d, index=False)
        print(f"Saved 3D results to: {out_3d}")

if __name__ == "__main__":
    main()
