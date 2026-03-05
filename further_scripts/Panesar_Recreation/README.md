# Panesar Model Baseline

This directory contains the training and evaluation scripts for the Panesar Model Baseline.

## 1. Setting Up the SUNRGBD Data

Before running any scripts, you need to ensure the SUNRGBD data is correctly positioned in the project directory.

The scripts expect the dataset to be located in a `dataset` directory at the root of the project. Specifically, the images and depth maps should be placed inside a `SUNRGBD` subfolder.

**Directory Structure:**
```
Knowledge_Distillation_for_Sensory_Substitution_in_Multimodal_Models/
├── dataset/
│   └── SUNRGBD/
│       ├── kv1/
│       ├── kv2/
│       ├── ... (rest of the dataset folders)
├── further_scripts/
│   └── Panesar_Recreation/
│       ├── panesar_model_baseline_train.py
│       ├── panesar_model_baseline_eval.py
│       └── ...
```

Make sure the path `h:\Knowledge_Distillation_for_Sensory_Substitution_in_Multimodal_Models\dataset\SUNRGBD` exists and contains your image and depth files. The dataset metadata itself is loaded automatically from the Hugging Face Hub (`shayekh00/VQA_SUNRGBD_v2`).

## 2. Training the Model (`panesar_model_baseline_train.py`)

The training script uses Optuna to optimize hyperparameters via Bayesian Optimization. It builds the required vocabulary mappings locally before starting.

To start training with the default parameters, simply run:

```bash
python panesar_model_baseline_train.py
```

### Available Arguments:
- `--epochs`: Number of maximum epochs to run per trial (default: `50`).
- `--n_trials`: Number of Optuna Bayesian trials to run (default: `10`).
- `--fusion_method`: Which RGB-D fusion mechanism to use. Options are `"hadamard"`, `"addition"`, `"maxpool"`, `"conv1d"`, or `"fusion_at_start"` (default: `"conv1d"`).
- `--patience`: Early stopping patience metrics for preventing overfitting (default: `5`).

**Example:**
```bash
python panesar_model_baseline_train.py --epochs 30 --n_trials 5 --fusion_method addition --patience 3
```

During training, intermediate models and the final optimal model will be saved inside the `models/` directory alongside the vocabulary JSON mappings (`word2idx.json`, `ans2idx.json`, etc.) which are required for evaluation.

## 3. Evaluating the Model (`panesar_model_baseline_eval.py`)

Once you have trained the model (or if you already have the pre-trained `.pth` weights and vocabulary files), you can evaluate it on the validation and test splits.

A basic run without specifying model weights will trigger an evaluation using randomly initialized weights (useful for debugging the data pipeline):

```bash
python panesar_model_baseline_eval.py
```

To run a proper evaluation, you must pass the path to your trained `.pth` weights using `--model_weights`.

### Available Arguments:
- `--model_weights`: Path to the pre-trained `.pth` model weights. (e.g., `models/panesar_model_optuna_conv1d_trial_0.pth`).
- `--fusion_method`: Must match the fusion method used during training! Options are `"hadamard"`, `"addition"`, `"maxpool"`, `"conv1d"`, or `"fusion_at_start"` (default: `"conv1d"`).
- `--max_rows`: Maximum number of rows to evaluate per dataset split, useful for quick tests (default: `None`, evaluates all rows).
- `--vocab_dir`: Path to the directory containing the vocabulary JSONs and config file generated during training (default: `models/`).

**Example:**
```bash
python panesar_model_baseline_eval.py --model_weights models/panesar_model_optuna_addition_trial_3.pth --fusion_method addition --max_rows 500
```

The evaluation script calculates the exact match accuracy for you and saves the predicted outputs as `.csv` files inside the `panesar_model_results/` directory for further analysis.
