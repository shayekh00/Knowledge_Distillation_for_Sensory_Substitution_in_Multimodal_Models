'''
Example use: 

python evaluation/onevisionv3/get_results.py

'''
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from metric import simple_accuracy_metric , neural_similarity_metric , compute_bert_stats, simple_accuracy_per_category

from torchmetrics.text.bert import BERTScore
# Load the CSV file
predictions_dir = "dataset/predictions/llava_onevision_checkpoint_double_trouble_phase2_-epoch=00-val_loss=0.0076.ckpt"
file_name = "_results_kd_model_type:depth_val_double_troublephase1.csv"


df = pd.read_csv(os.path.join(predictions_dir, file_name))


print("File loaded:", file_name)    
# Extract predictions (model answers) & references (ground truth answers)
predictions = df["Model_Answer"].tolist()
references = df["Answers"].tolist()

#Compute the metrics
category_accuracies = simple_accuracy_per_category(df)
print("Simple Accuracy per Category:", category_accuracies)


simple_accuracy = simple_accuracy_metric(predictions, references)
print("Simple Accuracy:", simple_accuracy)
neural_similarity = neural_similarity_metric(predictions, references)

print("Neural Similarity:", neural_similarity)

