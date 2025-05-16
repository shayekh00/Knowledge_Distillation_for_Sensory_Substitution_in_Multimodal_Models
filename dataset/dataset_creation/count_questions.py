import json
import numpy as np
import os
import cv2
from tqdm import tqdm
import pandas as pd
from utils import read_paths, get_object_name_list,find_most_prominent_object
# from dataset.question_generation2.utils import find_most_prominent_object
from collections import Counter, defaultdict
from post_process import process_text_only
import inflect

# Define the directory containing the file paths
directory_path = "dataset/SUNRGBD_Dataset/"
os.chdir(directory_path)

# Paths to your files - these need to be updated according to your dataset location
count_questions = []
count_answers = []
path_of_the_images = []
path_of_the_depth = []
item_ids = []

total_item_counts = defaultdict(int)
MAX_COUNT_PER_ITEM = 1000

def generate_questions_answers(item_counts, image_path, depth_path):
    for item, count in item_counts.items():
        # #Limit the number of questions per item
        if total_item_counts[item] < MAX_COUNT_PER_ITEM:
            question = f"How many {item}s are there?"
            count_questions.append(question)
            count_answers.append(count)
            path_of_the_images.append(image_path)
            path_of_the_depth.append(depth_path)
            total_item_counts[item] += 1

def generate_questions_answers_for_one_object(most_prominent_object_count, most_prominent_object_name, 
                                    most_prominent_object_name_processed, image_path, depth_path ,data_counter
            ):

    if total_item_counts[most_prominent_object_name] < MAX_COUNT_PER_ITEM:
        question = f"How many {most_prominent_object_name_processed}s are there?"
        count_questions.append(question)
        count_answers.append(most_prominent_object_count)
        path_of_the_images.append(image_path)
        path_of_the_depth.append(depth_path)
        item_ids.append(data_counter)
        # print(f"Question: {question}")
        # print(f"Answer: {most_prominent_object_count}")
        
        total_item_counts[most_prominent_object_name] += 1

def main():

    data_split = "validation"

    image_paths = read_paths("splits_output_paths/"+ data_split +"/all_rgb.txt")
    depth_paths = read_paths("splits_output_paths/"+ data_split +"/all_depth.txt")
    annotation_paths = read_paths("splits_output_paths/"+ data_split +"/annotations.txt")

    error_counter = 0
    data_counter = 1
    
    p = inflect.engine()

    with tqdm(total=len(image_paths)) as pbar:
        for image_path, depth_path, annotation_path in (
            zip(image_paths, depth_paths, annotation_paths)        ):
            try:
                with open(annotation_path, "r") as file:
                    annotation_data = json.load(file)

                object_names = get_object_name_list(annotation_data)
                # print(object_names)
                # Count occurrences of each item
                
                most_prominent_object_name = find_most_prominent_object(annotation_data)
                # print(most_prominent_object_name)
                most_prominent_object_name_processed = process_text_only(most_prominent_object_name)

                new_all_objects = []
                    
                for one_object in object_names:
                    processed_one_object = process_text_only(one_object)
                    new_all_objects.append(processed_one_object)

                item_counts = Counter(new_all_objects)
                most_prominent_object_count = item_counts[most_prominent_object_name_processed]
                most_prominent_object_count_in_words = p.number_to_words(most_prominent_object_count)
                
                
                # generate_questions_answers(item_counts, image_path, depth_path)
                generate_questions_answers_for_one_object(
                    most_prominent_object_count_in_words,most_prominent_object_name, most_prominent_object_name_processed, image_path, depth_path,data_counter
                )  

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
            "Questions": count_questions,
            "Answers": count_answers,
            "Image_Path": path_of_the_images,
            "Depth_Path": path_of_the_depth,
            "Question_Type": "Count"
        }
    )
    
    
    df.to_csv("SUNRGBD/csv_data/individual_datasets/"+ data_split +"/count_qa.csv", index=False)
    
    print(f"Number of errors: {error_counter}")
    print(f"Number of data total: {data_counter}")
    # print(f"Number of errors: {error_counter}")


if __name__ == "__main__":
    main()
