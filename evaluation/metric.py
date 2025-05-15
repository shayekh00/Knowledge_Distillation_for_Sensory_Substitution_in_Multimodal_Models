import spacy
import torch
from tqdm import tqdm
import pandas as pd
# Load the spaCy model
nlp = spacy.load("en_core_web_md")

def simple_accuracy_metric(predictions, references):
    """
    Computes accuracy by checking if each prediction matches its corresponding reference.
    Uses spaCy lemmatization for normalization.
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and References must have the same length.")

    correct_matches = 0
    total_items = len(predictions)

    for pred, ref in (zip(predictions, references)):
        # print("Outer Loop")
        # print(pred, ref)
        try:
            # Normalize prediction and reference using lemmatization
            pred_tokens = {token.lemma_.lower() for token in nlp(pred)}
            ref_tokens = {token.lemma_.lower() for token in nlp(ref)}
            # print(pred_tokens, ref_tokens)
            # Check if the reference is exactly in the prediction
            if ref_tokens == pred_tokens:
                # print("Inner Loop")
                correct_matches += 1
        except Exception:
            continue  # Skip if there's an issue with processing

    # Compute accuracy as the percentage of correctly matched predictions
    return correct_matches / total_items if total_items > 0 else 0.0

def neural_similarity_metric(predictions, references):
    """
    Compute the neural similarity between predictions and references
    using spaCy embeddings, comparing them one-to-one.
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and References must have the same length.")

    similarities = []

    for pred, ref in tqdm(zip(predictions, references), total=len(predictions), desc="Computing Neural Similarity"):
        try:
            pred_doc = nlp(pred)
            ref_doc = nlp(ref)
            similarity = pred_doc.similarity(ref_doc)
            similarities.append(similarity)
        except Exception:
            similarities.append(0.0)  # Default to 0 similarity if an error occurs

    # Return the mean similarity
    return sum(similarities) / len(similarities) if similarities else 0.0

def compute_bert_stats(bert_result):
    """
    Compute the mean and standard deviation of BERTScore precision, recall, and F1.
    """
    stats = {}
    
    for key in ['precision', 'recall', 'f1']:
        tensor_values = bert_result[key]
        stats[f"{key}_mean"] = torch.mean(tensor_values).item()  # Compute mean
        stats[f"{key}_std"] = torch.std(tensor_values).item()    # Compute standard deviation
    
    return stats




def simple_accuracy_per_category(df):
    """
    Computes simple accuracy for each question category (Question_Type) using the existing simple_accuracy_metric.
    """
    category_accuracies = {}
    
    # Group by Question_Type
    grouped = df.groupby("Question_Type")
    
    for category, group in grouped:
        predictions = group["Model_Answer"].tolist()
        references = group["Answers"].tolist()
        
        # Compute simple accuracy for the category
        accuracy = simple_accuracy_metric(predictions, references)
        category_accuracies[category] = accuracy
    
    # Sort categories by accuracy values in descending order
    sorted_accuracies = dict(sorted(category_accuracies.items(), key=lambda item: item[1], reverse=False))
    
    return sorted_accuracies

def neural_similarity_per_category(df):
    """
    Computes neural similarity for each question category (Question_Type) using the existing neural_similarity_metric.
    """
    category_similarities = {}
    
    # Group by Question_Type
    grouped = df.groupby("Question_Type")
    
    for category, group in grouped:
        predictions = group["Model_Answer"].tolist()
        references = group["Answers"].tolist()
        
        # Compute neural similarity for the category
        similarity = neural_similarity_metric(predictions, references)
        category_similarities[category] = similarity
    
    # Sort categories by similarity values in descending order
    sorted_similarities = dict(sorted(category_similarities.items(), key=lambda item: item[1], reverse=False))
    
    return sorted_similarities