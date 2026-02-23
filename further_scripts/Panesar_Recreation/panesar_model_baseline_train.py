import os
import sys
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset
import torchvision.transforms as transforms
import numpy as np

# Set up paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SYS_PATH = os.path.join(PROJECT_ROOT, 'inference')
MODELS_PATH = os.path.join(BASE_DIR, 'models')
sys.path.append(SYS_PATH)
sys.path.append(MODELS_PATH)

from inference_utils import get_image_paths, load_environment
from vqa_sunrgbd_model import VQASUNRGBDModel
import re

def normalize_answer(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = " ".join(text.split())
    return text

class TokenizerBuilder:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx = 2
        
    def fit(self, texts):
        for text in texts:
            words = str(text).lower().replace("?", "").replace(".", "").replace(",", "").split()
            for w in words:
                if w not in self.word2idx:
                    self.word2idx[w] = self.idx
                    self.idx += 1

    def encode(self, text, max_len=15):
        words = str(text).lower().replace("?", "").replace(".", "").replace(",", "").split()
        seq = [self.word2idx.get(w, self.word2idx["<UNK>"]) for w in words]
        if len(seq) < max_len:
            seq = seq + [0] * (max_len - len(seq))
        else:
            seq = seq[:max_len]
        return seq

def build_and_save_vocabs(train_data, vocab_dir, max_len=15):
    """
    Build vocabulary mappings from training split and save to JSON.
    """
    tokenizer = TokenizerBuilder()
    ans2idx = {}
    idx2ans = {}
    ans_idx = 0
    
    questions = []
    
    print("Building vocabulary from train split...")
    for row in tqdm(train_data, desc="Parsing train data"):
        questions.append(row.get("Questions", ""))
        ans = normalize_answer(row.get("Answers", ""))
        if ans not in ans2idx:
            ans2idx[ans] = ans_idx
            idx2ans[ans_idx] = ans
            ans_idx += 1
            
    tokenizer.fit(questions)
    
    # Save to JSON
    os.makedirs(vocab_dir, exist_ok=True)
    with open(os.path.join(vocab_dir, "word2idx.json"), "w") as f:
        json.dump(tokenizer.word2idx, f)
    
    idx2word = {v: k for k, v in tokenizer.word2idx.items()}
    with open(os.path.join(vocab_dir, "idx2word.json"), "w") as f:
        json.dump(idx2word, f)
        
    with open(os.path.join(vocab_dir, "ans2idx.json"), "w") as f:
        json.dump(ans2idx, f)
        
    with open(os.path.join(vocab_dir, "idx2ans.json"), "w") as f:
        json.dump(idx2ans, f)
        
    with open(os.path.join(vocab_dir, "config.json"), "w") as f:
        json.dump({"max_len": max_len}, f)
        
    return tokenizer.word2idx, ans2idx

def load_depth_as_rgb(depth_path):
    depth_img = Image.open(depth_path)
    if depth_img.mode == 'I;16':
        depth_array = np.array(depth_img, dtype=np.uint16)
        depth_array = (depth_array / 65535.0 * 255.0).astype(np.uint8)
        depth_img = Image.fromarray(depth_array, mode='L')
    else:
        depth_img = depth_img.convert("L")
    rgb_depth_img = Image.merge("RGB", (depth_img, depth_img, depth_img))
    return rgb_depth_img

class VQADataset(Dataset):
    def __init__(self, data_split, word2idx, ans2idx, root_data_dir, max_len=15):
        self.data = list(data_split)
        self.word2idx = word2idx
        self.ans2idx = ans2idx
        self.root_data_dir = root_data_dir
        self.max_len = max_len
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def encode_text(self, text):
        words = str(text).lower().replace("?", "").replace(".", "").replace(",", "").split()
        unk_idx = self.word2idx.get("<UNK>", 1)
        seq = [self.word2idx.get(w, unk_idx) for w in words]
        if len(seq) < self.max_len:
            seq = seq + [0] * (self.max_len - len(seq))
        else:
            seq = seq[:self.max_len]
        return seq

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        question = row.get("Questions", "")
        answer = normalize_answer(row.get("Answers", ""))
        
        image_path_suffix = row.get("Image_Path")
        depth_path_suffix = row.get("Depth_Path")
        
        rgb_path, depth_path = get_image_paths(image_path_suffix, depth_path_suffix, self.root_data_dir)
        
        if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
            # In training, if files are missing, just return the next or standard null (we should ideally pre-filter)
            return self.__getitem__((idx + 1) % len(self.data))
            
        # Modalities
        rgb_img = Image.open(rgb_path).convert("RGB")
        depth_img = load_depth_as_rgb(depth_path)
        
        rgb_tensor = self.transform(rgb_img)
        depth_tensor = self.transform(depth_img)
        
        q_tokenized = self.encode_text(question)
        q_tensor = torch.tensor(q_tokenized, dtype=torch.long)
        
        # Ans
        ans_idx = self.ans2idx.get(answer, 0)
        ans_tensor = torch.tensor(ans_idx, dtype=torch.long)
        
        return rgb_tensor, depth_tensor, q_tensor, ans_tensor

def main():
    import optuna
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--n_trials", type=int, default=10, help="Number of Optuna bayesian trials")
    parser.add_argument("--fusion_method", type=str, default="conv1d")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    args = parser.parse_args()
    
    ROOT_DATA_DIR = os.path.join(PROJECT_ROOT, "dataset")
    dataset_repo = "shayekh00/VQA_SUNRGBD_v2"
    print(f"Loading dataset: {dataset_repo}")
    dataset = load_dataset(dataset_repo)
    
    if "train" not in dataset or "validation" not in dataset:
        print("Dataset must have 'train' and 'validation' splits!")
        return
        
    word2idx, ans2idx = build_and_save_vocabs(dataset["train"], MODELS_PATH, max_len=15)
    
    vocab_size = len(word2idx)
    num_classes = len(ans2idx)
    print(f"Vocab size: {vocab_size}, Output classes: {num_classes}")
    
    train_dataset = VQADataset(dataset["train"], word2idx, ans2idx, ROOT_DATA_DIR)
    val_dataset = VQADataset(dataset["validation"], word2idx, ans2idx, ROOT_DATA_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    def objective(trial):
        # Bayesian optimization parameter space
        lr = trial.suggest_float("lr", 1e-4, 2.0, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        model = VQASUNRGBDModel(vocab_size=vocab_size, num_classes=num_classes, fusion_method=args.fusion_method)
        model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        # The paper specifies training using Adadelta, which is implemented below
        optimizer = optim.Adadelta(model.parameters(), lr=lr)
        
        best_val_loss = float('inf')
        epochs_no_improve = 0
        save_path = os.path.join(MODELS_PATH, f"panesar_model_optuna_{args.fusion_method}_trial_{trial.number}.pth")
        
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0.0
            
            loop = tqdm(train_loader, desc=f"Trial {trial.number} Epoch {epoch+1}/{args.epochs}")
            for rgb_t, depth_t, q_t, ans_t in loop:
                rgb_t, depth_t, q_t, ans_t = rgb_t.to(device), depth_t.to(device), q_t.to(device), ans_t.to(device)
                
                optimizer.zero_grad()
                logits = model(rgb_t, depth_t, q_t)
                loss = criterion(logits, ans_t)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                loop.set_postfix(loss=loss.item())
                
            train_loss /= len(train_loader)
            
            # Validation
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for rgb_t, depth_t, q_t, ans_t in val_loader:
                    rgb_t, depth_t, q_t, ans_t = rgb_t.to(device), depth_t.to(device), q_t.to(device), ans_t.to(device)
                    logits = model(rgb_t, depth_t, q_t)
                    loss = criterion(logits, ans_t)
                    val_loss += loss.item()
                    
                    preds = torch.argmax(logits, dim=1)
                    correct += (preds == ans_t).sum().item()
                    total += ans_t.size(0)
                    
            val_loss /= len(val_loader)
            val_acc = correct / total if total > 0 else 0
            
            print(f"Trial {trial.number} Epoch {epoch+1} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), save_path)
            else:
                epochs_no_improve += 1
                
            # Report intermediate objective value to Optuna 
            trial.report(val_loss, epoch)
            # Handle pruning based on the intermediate value
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
                
            # Early Stopping with predefined Patience
            if epochs_no_improve >= args.patience:
                print(f"Early stopping triggered during trial! Patience metric of {args.patience} reached without val improvement.")
                break
                
        return best_val_loss

    study = optuna.create_study(direction="minimize")
    print(f"Starting Bayesian Optimization with {args.n_trials} trials... Target: CrossEntropy Val Loss")
    study.optimize(objective, n_trials=args.n_trials)
    
    print("\n========== OPTIMIZATION COMPLETE ==========")
    print("Best trial found by Bayesian Search:")
    best_trial = study.best_trial
    print(f"  Lowest Val Loss: {best_trial.value}")
    print("  Best Hyperparameters: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
        
    print(f"The best weights have been locally saved at: {MODELS_PATH}/panesar_model_optuna_{args.fusion_method}_trial_{best_trial.number}.pth")

if __name__ == "__main__":
    main()
