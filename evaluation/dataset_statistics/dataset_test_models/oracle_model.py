import sys
import os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

ROOT_DATA_DIR = XXXX

def oracle_model(df, index):
    """
    Get the question and corresponding answer from the dataset given the index.

    Args:
        df (pandas.DataFrame): The dataset loaded as a DataFrame.
        index (int): The index of the item to retrieve.

    Returns:
        tuple: A tuple containing the question and the answer.
    """
    if index < 0 or index >= len(df):
        raise IndexError("Index out of range")

    # Retrieve the row at the given index
    row = df.iloc[index]

    # Extract the question and answer
    question = row['Questions']
    answer = row['Answers']

    return question, answer


if __name__ == "__main__":

    csv_file_path = os.path.join(ROOT_DATA_DIR, "SUNRGBD/csv_data", "train_dataset.csv")
    df = pd.read_csv(csv_file_path)

    # Test the function with an index
    test_index = 10
    question, answer = oracle_model(df, test_index)
    
    print(f"Question at index {test_index}: {question}")
    print(f"Answer at index {test_index}: {answer}")

