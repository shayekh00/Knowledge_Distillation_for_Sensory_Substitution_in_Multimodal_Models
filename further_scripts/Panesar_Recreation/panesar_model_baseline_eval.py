import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset
import torchvision.transforms as transforms

# Set up paths to import inference_utils and models
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SYS_PATH = os.path.join(PROJECT_ROOT, 'inference')
MODELS_PATH = os.path.join(BASE_DIR, 'models')
sys.path.append(SYS_PATH)
sys.path.append(MODELS_PATH)

from inference_utils import get_image_paths, load_environment
from vqa_sunrgbd_model import VQASUNRGBDModel
import re

class SimpleTokenizer:
    """
    A simple tokenizer to encode text questions into integer sequences,
    using a pre-loaded vocabulary.
    """
    def __init__(self, word2idx):
        self.word2idx = word2idx
        
    def encode(self, text, max_len=15):
        words = str(text).lower().replace("?", "").replace(".", "").replace(",", "").split()
        # Unknown words must map to the "<UNK>" index from word2idx
        unk_idx = self.word2idx.get("<UNK>", 1)
        seq = [self.word2idx.get(w, unk_idx) for w in words]
        if len(seq) < max_len:
            seq = seq + [0] * (max_len - len(seq))
        else:
            seq = seq[:max_len]
        return seq

def load_vocabularies_and_config(vocab_dir):
    """
    Load fixed vocabularies and configurations used during training.
    Raises FileNotFoundError if any required JSON file is missing.
    NO dynamic rebuilding happens here.
    """
    required_files = ["word2idx.json", "idx2word.json", "ans2idx.json", "idx2ans.json", "config.json"]
    loaded_data = {}
    
    for filename in required_files:
        filepath = os.path.join(vocab_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Required vocabulary/config file missing: {filepath}. "
                                    "Inference strictly requires pre-built mappings from training.")
        with open(filepath, 'r') as f:
            loaded_data[filename] = json.load(f)
            
    # json loads integer keys as strings, convert idx2word and idx2ans keys back to int
    idx2word = {int(k): v for k, v in loaded_data["idx2word.json"].items()}
    idx2ans = {int(k): v for k, v in loaded_data["idx2ans.json"].items()}
    
    # Extract max_len from config.json
    max_len = loaded_data["config.json"].get("max_len", 15)
    
    return (
        loaded_data["word2idx.json"],
        idx2word,
        loaded_data["ans2idx.json"],
        idx2ans,
        max_len
    )

def load_depth_as_rgb(depth_path):
    """
    Loads a depth image and converts it into a 3-channel (RGB) format
    as required by the VGG16 model, properly scaling depth values
    without turning it into a 3D XYZ map.
    """
    # Load raw depth image; commonly 16-bit
    depth_img = Image.open(depth_path)
    if depth_img.mode == 'I;16':
        depth_array = np.array(depth_img, dtype=np.uint16)
        # Normalize to [0, 255] by scaling
        # Assuming maximum depth possible is roughly the 16-bit max space, or you could do min-max normalization if per-image scaling was done in training.
        # Fixed scaling is generally safer for inference consistency to absolute metric distances.
        depth_array = (depth_array / 65535.0 * 255.0).astype(np.uint8)
        depth_img = Image.fromarray(depth_array, mode='L')
    else:
        # If it's already 8-bit or standard RGB, convert to grayscale first
        depth_img = depth_img.convert("L")
    
    # Repeat the single channel 3 times to create a standard RGB image format shape
    # required for models trained on ImageNet (VGG16)
    rgb_depth_img = Image.merge("RGB", (depth_img, depth_img, depth_img))
    return rgb_depth_img

def normalize_answer(text):
    """
    Standardize answer format: lowercase, remove punctuation, and strip whitespace.
    This corresponds to what standard VQA evaluation demands for exact matching.
    """
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = " ".join(text.split())
    return text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_rows", type=int, default=None, help="Max rows to process per split")
    parser.add_argument("--model_weights", type=str, default=None, help="Path to pre-trained model weights (.pth)")
    parser.add_argument("--fusion_method", type=str, default="conv1d", choices=["hadamard", "addition", "maxpool", "conv1d", "fusion_at_start"], help="Which RGB-D fusion method to use.")
    parser.add_argument("--vocab_dir", type=str, default=MODELS_PATH, help="Directory containing vocabulary JSONs and config.json")
    args = parser.parse_args()
    
    ROOT_DATA_DIR = os.path.join(PROJECT_ROOT, "dataset")
    
    dataset_repo = "shayekh00/VQA_SUNRGBD_v2"
    print(f"Loading dataset: {dataset_repo}")
    dataset = load_dataset(dataset_repo)
    
    # Image preprocessing as defined in the paper (VGG-16 standard preprocessing)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load fixed vocabularies and configuration from JSON files
    print(f"Loading vocabularies and config from {args.vocab_dir}...")
    try:
        word2idx, idx2word, ans2idx, idx2ans, max_len = load_vocabularies_and_config(args.vocab_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
        
    tokenizer = SimpleTokenizer(word2idx)
    vocab_size = len(word2idx)
    num_classes = len(ans2idx)
    
    print(f"Loaded config -> max_len: {max_len}")
    print(f"Loaded Vocab size: {vocab_size}, Output classes: {num_classes}")
    
    # Instantiate PyTorch Component 
    model = VQASUNRGBDModel(vocab_size=vocab_size, num_classes=num_classes, fusion_method=args.fusion_method)
    
    if args.model_weights and os.path.exists(args.model_weights):
        print(f"Loading weights from {args.model_weights}")
        model.load_state_dict(torch.load(args.model_weights, map_location="cpu"))
    else:
        print("Warning: No model weights provided. Running inference with randomly initialized Question/MLP weights for demonstration!")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    model.eval()
    
    splits = ["validation", "test"]
    output_dir = os.path.join(BASE_DIR, "panesar_model_results")
    os.makedirs(output_dir, exist_ok=True)
    
    for split in splits:
        if split not in dataset:
            continue
            
        print(f"\n========== Processing split: {split.upper()} ==========")
        split_data = dataset[split]
        
        data_list = []
        predictions = []
        references = []
        
        for i, row in tqdm(enumerate(split_data), total=args.max_rows if args.max_rows else len(split_data)):
            if args.max_rows and i >= args.max_rows:
                break
                
            question_id = row.get("Question_Id")
            question = row.get("Questions")
            question_type = row.get("Question_Type")
            
            # Ground truth answer, normalized for evaluation
            answers = normalize_answer(row.get("Answers", ""))
            
            image_path_suffix = row.get("Image_Path")
            depth_path_suffix = row.get("Depth_Path")
            
            if not depth_path_suffix or not image_path_suffix:
                continue
                
            # Identical resolution procedure to inference.py 
            rgb_path, depth_path = get_image_paths(
                image_path_suffix, 
                depth_path_suffix, 
                ROOT_DATA_DIR
            )
            
            if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
                # Optionally warn and skip if files not locally downloaded/matching paths
                continue
                
            try:
                # Modality 1: RGB 
                rgb_img = Image.open(rgb_path).convert("RGB")
                
                # Modality 2: Depth (as a 3-channel depth map)
                depth_img = load_depth_as_rgb(depth_path)
                
                # Push through transforms and unsqueeze batch dimension
                rgb_tensor = transform(rgb_img).unsqueeze(0).to(device)
                depth_tensor = transform(depth_img).unsqueeze(0).to(device)
                
                # Question textual tokenization
                q_tokenized = tokenizer.encode(question, max_len=max_len)
                q_tensor = torch.tensor([q_tokenized], dtype=torch.long).to(device)
                
                # PyTorch Inference
                with torch.no_grad():
                    logits = model(rgb_tensor, depth_tensor, q_tensor)
                    pred_idx = torch.argmax(logits, dim=1).item()
                    
                model_final_answer = idx2ans.get(pred_idx, "UNKNOWN")
                
                data_list.append({
                    'Question_Id': question_id,
                    'Questions': question,
                    'Question_Type': question_type,
                    'Answers': answers,
                    'Model_Answer': model_final_answer
                })
                
                predictions.append(model_final_answer)
                references.append(answers)
                
            except Exception as e:
                print(f"[Error] Failed processing ID {question_id}. Error: {e}")
                
        # Calculate Basic Accuracy Metrics
        if len(predictions) > 0 and len(references) > 0:
            exact_matches = sum(1 for p, r in zip(predictions, references) if p == r)
            accuracy = exact_matches / len(predictions)
            print(f"\nExact Match Accuracy ({split}): {accuracy * 100:.2f}% ({exact_matches}/{len(predictions)})")
            
        # Save results out to Pandas DF
        if len(data_list) > 0:
            results_df = pd.DataFrame(data_list)
            print(f"\nResults DataFrame ({split}):")
            print(results_df.head())
            
            out_csv = os.path.join(output_dir, f"panesar_model_{split}_results.csv")
            results_df.to_csv(out_csv, index=False)
            print(f"Saved results to: {out_csv}")

if __name__ == "__main__":
    main()
