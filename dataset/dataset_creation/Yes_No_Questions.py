import json
import numpy as np
import os
import cv2
from tqdm import tqdm
import pandas as pd
from utils import read_paths, clean_and_dedupe, get_object_name_list ,find_most_prominent_object,find_object_index
from collections import Counter
import random
import string
from post_process import process_text_only

# Define the directory containing the file paths
directory_path = "dataset/SUNRGBD_Dataset/"
# Change the current working directory to the specified directory
os.chdir(directory_path)

yes_no_questions = []
yes_no_answers = []
path_of_the_images = []
path_of_the_depth = []
item_ids = []

def append_list(question, answer, image_path, depth_path,data_counter):
    yes_no_questions.append(question)
    yes_no_answers.append(answer)
    path_of_the_images.append(image_path)
    path_of_the_depth.append(depth_path)
    item_ids.append(data_counter)
    


def generate_questions_answers(
    item_counts, unique_items_list_cleaned, image_path, depth_path
):
    item_names = list(item_counts.keys())

    values_not_in_item_names = [
        item for item in unique_items_list_cleaned if item not in item_names
    ]

    for item in item_counts.items():
        question = f"Is there any {item[0]}?"
        answer = "Yes"

        append_list(question, answer, image_path, depth_path)

        random_value = random.choice(values_not_in_item_names)
        question = f"Is there no {random_value}?"
        answer = "No"

        append_list(question, answer, image_path, depth_path)

def generate_questions_answers_for_one_object(
    most_prominent_object_count,most_prominent_object_name, unique_items_list_cleaned, image_path, depth_path, data_counter
):
    # print("Unique List",unique_items_list_cleaned)
    # print("Most Prominent Item Count", most_prominent_object_count)
    # print("Most Prominent Item Name",most_prominent_object_name)




    question = f"Is there any {most_prominent_object_name}?"
    answer = "yes"

    append_list(question, answer, image_path, depth_path, data_counter)

    filtered_items = {item for item in unique_items_list_cleaned if item != most_prominent_object_name}
    random_item = random.choice(list(filtered_items))

    question = f"Is there any {random_item}?"
    answer = "no"

    append_list(question, answer, image_path, depth_path, data_counter)

def main():
    # image_paths = read_paths("all_rgb.txt")
    # depth_paths = read_paths("all_depth.txt")
    # annotation_paths = read_paths("annotations.txt")


    data_split = "validation"
    image_paths = read_paths("splits_output_paths/"+ data_split +"/all_rgb.txt")
    depth_paths = read_paths("splits_output_paths/"+ data_split +"/all_depth.txt")
    annotation_paths = read_paths("splits_output_paths/"+ data_split +"/annotations.txt")
    
    unique_items = pd.read_excel("unique_items.xlsx", sheet_name="Sheet1")
    unique_items_list = unique_items.values.tolist()
    unique_items_list = [item for sublist in unique_items_list for item in sublist]
    unique_items_list_cleaned = clean_and_dedupe(unique_items_list)

    error_counter = 0
    data_counter = 1

    with tqdm(total=len(image_paths)) as pbar:
        for image_path, depth_path, annotation_path in (
            zip(image_paths, depth_paths, annotation_paths)
        ):
        
            try:
                with open(annotation_path, "r") as file:
                    annotation_data = json.load(file)

                    


                object_names = get_object_name_list(annotation_data)
                # Count occurrences of each item
                item_counts = Counter(object_names)
                most_prominent_object_name = find_most_prominent_object(annotation_data)
                most_prominent_object_count = item_counts[most_prominent_object_name]
                
                most_prominent_object_name = process_text_only(most_prominent_object_name)
                generate_questions_answers_for_one_object(
                    most_prominent_object_count,most_prominent_object_name, unique_items_list_cleaned, image_path, depth_path, data_counter
                )
                # item_ids.append(data_counter)
                
                

            except Exception as e:
                error_counter += 1
                # print(e)
                continue

            data_counter += 1

            pbar.set_postfix({"errors": error_counter, "processed": data_counter})
            pbar.update(1)

    # Create DataFrame
    df = pd.DataFrame(
        {   "IDs": item_ids,
            "Questions": yes_no_questions,
            "Answers": yes_no_answers,
            "Image_Path": path_of_the_images,
            "Depth_Path": path_of_the_depth,
            "Question_Type": "Yes/No"
        }
    )

    half_df = df.iloc[:len(df) // 2]

    half_df.to_csv("SUNRGBD/csv_data/individual_datasets/"+ data_split +"/yes_no_qa.csv", index=False)



    print(f"Number of errors: {error_counter}")
    print(f"Number of data total: {data_counter}")



if __name__ == "__main__":
    main()
