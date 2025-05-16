import pandas as pd
import os
from post_process import process_answers_column

# dataset/question_generation2/post_process.py
# Get the current working directory
current_path = os.getcwd()
# Print the current working directory
print("Current working directory:", current_path)



# dataset/SUNRGBD_Dataset/SUNRGBD/csv_data/individual_datasets/new
# Specify the directory containing the CSV files
directory = "dataset/SUNRGBD_Dataset/SUNRGBD/csv_data"
individual_ds_directory = directory+ "/individual_datasets/new/"

# Create an empty DataFrame to hold data from all CSV files
combined_df = pd.DataFrame(columns=["Questions", "Answers", "Image_Path", "Depth_Path","Question_Type"])

# Loop through all files in the specified directory
for filename in os.listdir(individual_ds_directory):
    if filename.endswith(".csv") and filename != 'final_dataset.csv':  # Ensure it is a CSV file
        # Construct the full file path
        file_path = os.path.join(individual_ds_directory, filename)

        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # Ensure the DataFrame contains the expected columns
        if set(["Questions", "Answers", "Image_Path", "Depth_Path","Question_Type"]).issubset(
            df.columns
        ):
            # Concatenate the current DataFrame to the combined DataFrame
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        else:
            print(
                f"File {filename} does not contain the required columns and will be skipped."
            )

# Replace occurrences of "grey" with "gray" in the 'Answers' column
combined_df['Answers'] = combined_df['Answers'].str.replace(r'\bgrey\b', 'gray', case=False, regex=True)

# Count initial number of rows
initial_row_count = combined_df.shape[0]

# Find rows containing the exact words "all" or "alls" in the 'Questions' or 'Answers' columns
all_rows = combined_df[
    combined_df["Questions"].str.contains(r'\b(all|alls)\b', case=False, na=False) |
    combined_df["Answers"].str.contains(r'\b(all|alls)\b', case=False, na=False)
]

# # Print rows containing "all"
# print("Rows containing the exact word 'all' in Questions or Answers column:")
# print(all_rows)

# Delete rows containing the exact word "all"
combined_df = combined_df.drop(all_rows.index).reset_index(drop=True)

# Count final number of rows and calculate rows deleted
final_row_count = combined_df.shape[0]
rows_deleted = initial_row_count - final_row_count

# Print the number of rows deleted
print(f"\nNumber of rows deleted: {rows_deleted}")


# Shuffle the DataFrame
shuffled_df = combined_df.sample(frac=1).reset_index(drop=True)
print(f"Combined dataset length: {len(shuffled_df)}")

# shuffled_df = process_answers_column(shuffled_df, combined_words)

# Calculate the indices for the split
total_size = len(shuffled_df)
train_end = int(0.7 * total_size)
test_end = int(0.9 * total_size)  # 0.7 + 0.2 = 0.9

# Split the dataset
train_df = shuffled_df[:train_end]
test_df = shuffled_df[train_end:test_end]
val_df = shuffled_df[test_end:]


# Add 'Id' column to each dataset
train_df.insert(0, 'Question_Id', range(1, len(train_df) + 1))  # Add Id starting from 1
test_df.insert(0, 'Question_Id', range(1, len(test_df) + 1))    # Add Id starting from 1
val_df.insert(0, 'Question_Id', range(1, len(val_df) + 1))      # Add Id starting from 1

# Print the shapes of the splits to verify
print(f"Training set: {train_df.shape}")
print(f"Testing set: {test_df.shape}")
print(f"Validation set: {val_df.shape}")

# Specify the paths for the output CSV files
ds_version = str(4)

train_file_path = directory + "/train_dataset" + ds_version + ".csv"
test_file_path = directory + "/test_dataset" + ds_version + ".csv"
val_file_path = directory + "/val_dataset" + ds_version + ".csv"


# Write the DataFrames to their respective CSV files
train_df.to_csv(train_file_path, index=False)
test_df.to_csv(test_file_path, index=False)
val_df.to_csv(val_file_path, index=False)

# Print confirmation messages
print(f"Training dataset has been saved to {train_file_path}")
print(f"Testing dataset has been saved to {test_file_path}")
print(f"Validation dataset has been saved to {val_file_path}")
