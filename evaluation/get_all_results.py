'''
Example usage: 
python evaluation/get_all_results.py

'''

import pandas as pd
import os
import sys
from metric import simple_accuracy_metric, neural_similarity_metric, compute_bert_stats, simple_accuracy_per_category, neural_similarity_per_category

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# Directory containing prediction files
predictions_dir = "dataset/predictions/"
results_file = "dataset/predictions/summary/results_summary.csv"

# Load existing results if available
if os.path.exists(results_file):
    existing_results = pd.read_csv(results_file)
    processed_files = set(existing_results["File_Name"].tolist())
else:
    existing_results = pd.DataFrame()
    processed_files = set()

all_results = []

# Retrieve and sort file names alphabetically
file_names = sorted(f for f in os.listdir(predictions_dir) if f.endswith(".csv") and f not in processed_files)

for file_name in file_names:
    file_path = os.path.join(predictions_dir, file_name)
    df = pd.read_csv(file_path)
    print(f"Processing file: {file_name}")
    
    predictions = df["Model_Answer"].tolist()
    references = df["Answers"].tolist()
    
    # Compute metrics
    simple_accuracy = simple_accuracy_metric(predictions, references)
    category_accuracies = simple_accuracy_per_category(df)
    neural_similarity = neural_similarity_metric(predictions, references)
    category_similarities = neural_similarity_per_category(df)
    
    # Flatten category-based results into string format
    category_accuracies_str = str(category_accuracies)
    category_similarities_str = str(category_similarities)

    row = {
        "File_Name": file_name,
        "Simple_Accuracy": simple_accuracy,
        "Simple_Accuracy_Per_Category": category_accuracies_str,
        "Neural_Similarity": neural_similarity,
        "Neural_Similarity_Per_Category": category_similarities_str
    }
    
    all_results.append(row)
    
# Convert results to DataFrame and sort alphabetically before merging
new_results_df = pd.DataFrame(all_results).sort_values(by=["File_Name"])

# Append new results to existing results
if not existing_results.empty:
    results_df = pd.concat([existing_results, new_results_df], ignore_index=True).sort_values(by=["File_Name"])
else:
    results_df = new_results_df

# Save results to CSV
results_df.to_csv(results_file, index=False)

print("Results saved to", results_file)
print("Done!")