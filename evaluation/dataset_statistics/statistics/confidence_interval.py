import os
import pandas as pd
import scipy.stats as stats
import math

def calculate_confidence_interval(data, confidence_level=0.95):
    """
    Calculate the confidence interval for a given dataset.
    :param data: A pandas Series or list of numerical data.
    :param confidence_level: The confidence level for the interval (default is 0.95).
    :return: A tuple containing the lower and upper bounds of the confidence interval.
    """
    mean = data.mean()
    std_dev = data.std()
    n = len(data)
    
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    margin_of_error = z_score * (std_dev / math.sqrt(n))
    
    return mean - margin_of_error, mean + margin_of_error

def calculate_proportion_confidence_interval(data, confidence_level=0.95):
    """
    Calculate the confidence interval for a proportion (binary data: Check = 1 or 2).
    :param data: A pandas Series containing binary data (1 = OK, 2 = Not OK).
    :param confidence_level: The confidence level for the interval (default is 0.95).
    :return: A tuple containing the lower and upper bounds of the confidence interval.
    """
    n = len(data)  # Total number of samples
    p_hat = (data == 2).mean()  # Proportion of "Not OK" (Check = 2)
    
    z_score = stats.norm.ppf((1 + confidence_level) / 2)  # Z-score for confidence level
    margin_of_error = z_score * math.sqrt((p_hat * (1 - p_hat)) / n)
    
    return max(0, p_hat - margin_of_error), min(1, p_hat + margin_of_error)  # Ensuring CI bounds are valid

if __name__ == "__main__":
    # Update the ROOT_DATA_DIR path as per your system
    ROOT_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
    directory_path = 'dataset/SUNRGBD_Dataset/SUNRGBD/csv_data/'
    # Path to the dataset file
    csv_file_path = os.path.join(
        directory_path,
        "test_dataset_check_confidence_interval.csv"
    )
    
    # Load the dataset
    try:
        df = pd.read_csv(csv_file_path)
        print("Dataset loaded successfully.")
    except FileNotFoundError:
        print(f"File not found: {csv_file_path}")
        exit()
    
    # Filter the 'Check' column
    if 'Check' not in df.columns:
        print("The 'Check' column is missing from the dataset.")
        exit()
    
    check_values = df['Check']
    
    # Calculate the 95% confidence interval
    confidence_interval = calculate_proportion_confidence_interval(check_values)
    print(f"95% Confidence Interval: {confidence_interval}")
