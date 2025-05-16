# Knowledge Distillation for Sensory Substitution in Multimodal Models

This project explores knowledge distillation techniques to transfer multimodal reasoning abilities from RGB-based teacher models to depth-based student models. It supports various distillation strategies implemented with PyTorch Lightning, evaluated on SUNRGBD.

---
![image](https://github.com/user-attachments/assets/dc0e884e-13ef-483a-91e8-da1cbd712ff5)


## 📁 Repository Structure

```
Knowledge_Distillation_for_Sensory_Substitution_in_Multimodal_Models/
├── dataset/
├── distillation/
│   ├── baseline_depth/
│   ├── baseline_rgb05b/
│   ├── baseline_rgb7b/
│   ├── knowledge_distillation7b_double_trouble/
│   │   ├── phase1/
│   │   ├── phase2/
│   │   └── phase3/
│   ├── knowledge_distillation7b_feature_based/
│   └── knowledge_distillation7b_logit_based/
├── evaluation/
│   └── onevisionv3/
│       └── sunrgbd/
├── utils.py
├── requirements.txt
├── .env
├── README.md
```

---

## 🎯 Project Goals
- Recreate a new version of the VQA-SUNRGBD dataset.
- Distill from a large RGB-based teacher (7B) to a lightweight depth-based student (0.5B)
- Enable multimodal understanding with depth data.
- Experiment with multiple distillation strategies: feature-based, logit-based, and hybrid
- Evaluate performance on VQA.

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/knowledge_distillation_for_sensory_substitution_in_multimodal_models.git
cd knowledge_distillation_for_sensory_substitution_in_multimodal_models
```

### 2. Install dependencies

```bash
conda create -n kd_env python=3.9
conda activate kd_env
pip install -r requirements.txt
```

### 3. Environment variables

Create a `.env` file at the root:

```env
ROOT_DATA_DIR=/absolute/path/to/your/data/
MAIN_ROOT_DATA_DIR=/absolute/path/to/your/home/
hf_token=your_huggingface_token
```

---

## 🏋️ Training

### Phase 1: Vision-level contrastive distillation

```bash
python distillation/knowledge_distillation7b_double_trouble/phase1/train_online_kd.py --batch_size 1 --max_epochs 10 --subset_percentage 1 --load_checkpoint

```

### Phase 2: Logit distillation via LoCa

```bash
python distillation/knowledge_distillation7b_double_trouble/phase2/train_online_kd.py --batch_size 1 --max_epochs 10 --subset_percentage 1 --load_checkpoint
```

### Phase 3: Optional Combined distillation (vision + logits)

```bash
python distillation/knowledge_distillation7b_double_trouble/phase3/train_online_kd.py --batch_size 1 --max_epochs 10 --subset_percentage 1 --load_checkpoint
```

---

## 🧪 Baseline Training

### RGB Student Baseline (0.5B)

```bash
python distillation/baseline_rgb05b/train.py --batch_size 2 --max_epochs 5 --subset_percentage 1 --augmentation --accumulate_grad_batches 32
```

### RGB Teacher (7B)

```bash
python distillation/baseline_rgb7b/train2.py --batch_size 2 --max_epochs 10 --subset_percentage 1 --augmentation --accumulate_grad_batches 32
```

### Depth-Only Baseline (0.5B)

```bash
python distillation/baseline_depth/train.py --batch_size 2 --max_epochs 5 --subset_percentage 1 --augmentation --accumulate_grad_batches 32
```

### 🛠️ Dataset Generation Steps

1. **Extract the initial data:**

    ```bash
    python dataset/dataset_creation/extract_data.py
    ```

2. **Generate the question datasets:**

    Run the following scripts:

    ```bash
    python dataset/dataset_creation/color_questions.py
    python dataset/dataset_creation/count_questions.py
    python dataset/dataset_creation/object_identification.py
    python dataset/dataset_creation/ProximityQuestion.py
    python dataset/dataset_creation/Yes_No_Questions.py
    ```

3. **Combine all individual datasets into a final dataset:**

    ```bash
    python dataset/dataset_creation/merge_all_csv.py
    ```


---

## 📊 Evaluation

Evaluate on SUNRGBD:

```bash
python evaluation/onevisionv3/sunrgbd/evaluate_onevision.py --ckpt_path path/to/checkpoint.ckpt

#Example: python evaluation/onevisionv3/evaluate_onevision.py --model_id llava-hf/llava-onevision-qwen2-0.5b-ov-hf --gts_type val --kd_model_type double_trouble --phase_no 1 --pixel_data_type depth --load_checkpoint --student_ckpt_path dummy_model-val_loss=6.1143.ckpt
```

---

## 🧰 Logging and Outputs

- **Checkpoints** saved in `checkpoints/` under each phase or baseline
- **TensorBoard logs** stored in `tensorboard_logs/`
- Final `.pt` model weights saved manually if specified

---

## 📚 References

- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [SUNRGBD Dataset](https://rgbd.cs.princeton.edu/)
- [LoCa: Logit Calibration](https://arxiv.org/abs/2409.04778)

---

## 🪪 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
