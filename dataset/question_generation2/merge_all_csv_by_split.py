import pandas as pd
import os
from post_process import process_answers_column  # Import if still needed
import re

# Get the current working directory
current_path = os.getcwd()
print("Current working directory:", current_path)

# Define base directory for individual datasets
ROOT_DATA_DIR = os.getenv("root_data_dir2")
base_directory = "/individual_datasets"
# Define split directories
split_directories = {
    "train": os.path.join(base_directory, "train"),
    "test": os.path.join(base_directory, "test"),
    "validation": os.path.join(base_directory, "validation"),
}

# Output file paths
output_file_paths = {
    "train": os.path.join(base_directory, "train_dataset.csv"),
    "test": os.path.join(base_directory, "test_dataset.csv"),
    "validation": os.path.join(base_directory, "val_dataset.csv"),
}


def balance_dataset_by_question_type(df, question_type_col='Question_Type', image_path_col='Image_Path'):
    """
    Balances a dataset to ensure equal representation of each Question_Type
    with one unique Image_Path per question type.

    Parameters:
        df (pd.DataFrame): The input dataframe to balance.
        question_type_col (str): The column name for question types. Default is 'Question_Type'.
        image_path_col (str): The column name for image paths. Default is 'Image_Path'.

    Returns:
        pd.DataFrame: A balanced dataframe.
    """
    # Determine the set of unique Image_Paths
    unique_image_paths = set(df[image_path_col])

    # Calculate the number of samples for each Question_Type
    no_of_data_of_each_question_type = len(unique_image_paths) // df[question_type_col].nunique()
    print(f"Number of samples for each question type: {no_of_data_of_each_question_type}")

    # Initialize an empty dataframe to store the balanced data
    balanced_df = pd.DataFrame()

    # Iterate over each Question_Type
    question_types = df[question_type_col].unique()

    image_path_series = df["Image_Path"].tolist()
    unique_image_paths = sorted(list(set(image_path_series)))
    

    for question_type in question_types:

        selected_image_paths = unique_image_paths[:no_of_data_of_each_question_type]
        unique_image_paths = [path for path in unique_image_paths if path not in selected_image_paths]

        question_type_rows = df[df[question_type_col] == question_type]

        selected_rows = question_type_rows[question_type_rows[image_path_col].isin(selected_image_paths)]

        balanced_df = pd.concat([balanced_df, selected_rows], ignore_index=True)

        
    return balanced_df



def balance_yes_no_question_type(df, question_type_col='Question_Type', answers_col='Answers'):
    """
    Halves the number of rows for 'Yes/No' Question_Type while ensuring
    equal distribution of 'yes' and 'no' answers.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        question_type_col (str): The column for Question_Type.
        answers_col (str): The column for Answers.
    
    Returns:
        pd.DataFrame: The modified DataFrame with balanced 'Yes/No' Question_Type.
    """
    # Filter rows with Question_Type as 'Yes/No'
    yes_no_df = df[df[question_type_col] == 'Yes/No']
    
    # Separate rows with answers 'yes' and 'no'
    yes_rows = yes_no_df[yes_no_df[answers_col].str.lower() == 'yes']
    no_rows = yes_no_df[yes_no_df[answers_col].str.lower() == 'no']
    
    # Calculate the target number of rows (half the current count)
    target_count = len(yes_no_df) // 2
    target_yes_count = target_count // 2
    target_no_count = target_count // 2
    
    # Sample rows to match the target count
    sampled_yes_rows = yes_rows.sample(n=target_yes_count, random_state=42)
    sampled_no_rows = no_rows.sample(n=target_no_count, random_state=42)
    
    # Concatenate the balanced rows
    balanced_yes_no_df = pd.concat([sampled_yes_rows, sampled_no_rows], ignore_index=True)
    
    # Exclude original Yes/No rows from the main DataFrame
    non_yes_no_df = df[df[question_type_col] != 'Yes/No']
    
    # Combine the balanced Yes/No rows with the rest of the DataFrame
    final_df = pd.concat([non_yes_no_df, balanced_yes_no_df], ignore_index=True)
    
    return final_df


# Function to process each split
def process_split(split_name, split_directory, output_file_path):
    print(f"\nProcessing {split_name} split...")

    # Create an empty DataFrame to hold data from all CSV files
    combined_df = pd.DataFrame(columns=["Questions", "Answers", "Image_Path", "Depth_Path", "Question_Type"])

    # Loop through all files in the split directory
    for filename in os.listdir(split_directory):
        if filename.endswith(".csv"):  # Ensure it is a CSV file
            # Construct the full file path
            file_path = os.path.join(split_directory, filename)

            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)

            # Ensure the DataFrame contains the expected columns
            if set(["Questions", "Answers", "Image_Path", "Depth_Path", "Question_Type"]).issubset(df.columns):
                # Concatenate the current DataFrame to the combined DataFrame
                combined_df = pd.concat([combined_df, df], ignore_index=True)
            else:
                print(f"File {filename} does not contain the required columns and will be skipped.")

    # Replace occurrences of "grey" with "gray" in the 'Answers' column
    combined_df['Answers'] = combined_df['Answers'].str.replace(r'\bgrey\b', 'gray', case=False, regex=True)

    # Word replacement dictionary (for exact matches)
    replacement_dict = {
        r'\bouchs\b': 'couches',
        r'\btouches\b': 'couches',
        r'\btouchs\b': 'couches',
        r'\bhairs\b': 'chairs',
        r'\bwhat\?\b': 'hat',
        r'\bdivided\b': 'divider',
        r'\bdivideds\b': 'divider',
        r'\bdesk tops\b': 'desktops',
        r'\bdesk top\b': 'desktops',

        r'\bots\b': 'tvs',
        r'\battress\b': 'matterss',
        r'\bchairperson\b': 'chair',
        r'\bwater purified\b': 'water purifier',
        r'\bconstable\b': 'table',

        r'\bloveseat\b': 'couch',
        r'\bmind fridge\b': 'mini fridge',
        r'\bchase\b': 'chair',
        r'\bhair\b': 'chair',
        r'\bso far\b': 'sofa',

        r'\btimes\b': 'tiles',
        r'\bfaiting\b': 'painting',
        r'\bshelling\b': 'ceiling',
        r'\bcomfortable\b': 'comforter',
        r'\bprotector screen\b': 'projector screen',
        r'\bcurrent\b': 'curtain',
        r'\bchart\b': 'trash',
        r'\batble\b': 'table',
        r'\bbacket\b': 'bucket',
        r'\bauricle\b': 'cubicle',
        r'\bpurified\b': 'purifier',
    }

    # Special case for "red" to only modify in the Questions column
    red_pattern = r'\bred\b'
    red_replacement = 'bed'

    # Special case for "what?" to only modify in the Answers column
    what_pattern = r'\bwhat\?\b'
    what_replacement = 'hat'

    # Apply replacements and log changes
    def replace_and_log(row):
        modified = False
        original_row = row.copy()  # Save the original row for logging
        
        # Replace "red" in Questions column only
        if pd.notna(row['Questions']):
            if re.search(red_pattern, row['Questions'], flags=re.IGNORECASE):
                row['Questions'] = re.sub(red_pattern, red_replacement, row['Questions'], flags=re.IGNORECASE)
                modified = True

        # Replace "what?" in Answers column only
        if pd.notna(row['Answers']):
            if re.search(what_pattern, row['Answers'], flags=re.IGNORECASE):
                row['Answers'] = re.sub(what_pattern, what_replacement, row['Answers'], flags=re.IGNORECASE)
                modified = True

        # Apply general replacements to both Questions and Answers
        for pattern, replacement in replacement_dict.items():
            if pd.notna(row['Questions']):
                if re.search(pattern, row['Questions'], flags=re.IGNORECASE):
                    row['Questions'] = re.sub(pattern, replacement, row['Questions'], flags=re.IGNORECASE)
                    modified = True
            if pd.notna(row['Answers']):
                if re.search(pattern, row['Answers'], flags=re.IGNORECASE):
                    row['Answers'] = re.sub(pattern, replacement, row['Answers'], flags=re.IGNORECASE)
                    modified = True

        return row

    # Apply the replace_and_log function to each row
    combined_df = combined_df.apply(replace_and_log, axis=1)

    # Count initial number of rows
    initial_row_count = combined_df.shape[0]

    # Remove rows containing specific patterns in 'Questions' or 'Answers' columns
    patterns_to_remove = r'wall\d+'  # Matches strings like "wall40", "wall22", etc.
    exact_match_to_remove = r'^i think$'  # Exact match for "i think"

    # Apply the filter to remove rows
    combined_df = combined_df[
        ~(
            combined_df["Questions"].str.contains(patterns_to_remove, case=False, na=False) |
            combined_df["Answers"].str.contains(patterns_to_remove, case=False, na=False) |
            combined_df["Questions"].str.match(exact_match_to_remove, case=False, na=False) |
            combined_df["Answers"].str.match(exact_match_to_remove, case=False, na=False)
        )
    ]
    
    print(f"Dataset length after removing rows with unwanted patterns: {len(combined_df)}")

    # Balance the dataset if it's for test or validation splits
    if split_name == "test" or split_name == "validation":
        balanced_df = balance_dataset_by_question_type(combined_df, question_type_col='Question_Type', image_path_col='Image_Path')
        balanced_df = balance_yes_no_question_type(balanced_df, question_type_col='Question_Type', answers_col='Answers')
    else:
        balanced_df = combined_df


    # Apply the function after processing the combined DataFrame
    

    # Add 'Question_Id' column to the dataset
    balanced_df.insert(0, 'Question_Id', range(1, len(balanced_df) + 1))  # Add Id starting from 1

    # Number of rows for each Question_Type
    question_type_counts = balanced_df['Question_Type'].value_counts()
    print("Number of rows for each Question_Type:")
    print(question_type_counts)

    # Write the processed DataFrame to a CSV file
    balanced_df.to_csv(output_file_path, index=False)
    print(f"{split_name.capitalize()} dataset has been saved to {output_file_path}")
    print(f"Number of rows in {split_name} dataset: {len(balanced_df)}")




# Process each split
for split_name, split_directory in split_directories.items():
    output_file_path = output_file_paths[split_name]
    process_split(split_name, split_directory, output_file_path)



