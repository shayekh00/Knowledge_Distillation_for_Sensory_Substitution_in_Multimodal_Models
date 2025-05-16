import numpy as np
from shapely.geometry import Polygon
import os
import json
from PIL import Image
from tqdm import tqdm
from collections import Counter
import pandas as pd
from utils import read_paths
from utils import read_paths ,find_most_prominent_object ,find_object_index ,find_object_polygon
from post_process import process_text_only
# ws value
ws = 1.3


def is_number(n):
    return isinstance(n, (int, float, complex))


# Function to calculate the area of a bounding box from polygon points
def calculate_bounding_box_area(polygon):
    if len(polygon) < 4:
        return 0

    poly = Polygon(polygon)
    return poly.area


# Function to calculate the average depth value (Z) from XYZ coordinates
def calculate_average_depth(obj):
    if not obj["XYZ"]:
        return float("inf")  # Consider objects with no depth as infinitely far away

    XYZ = obj["XYZ"]
    # z_values = [point[2] for point in XYZ]
    z_values = []
    for point in XYZ:
        try:
            if is_number(point[2]):
                z_values.append(point[2])
        except:
            continue

    return np.mean(z_values)


def remove_unwanted_objects(data, unwanted_names):
    # Collect indices of objects to remove
    objects_to_remove = []
    
    # Identify objects to remove based on their names
    for idx, obj in enumerate(data["objects"]):
        if any(unwanted_name in obj["name"].lower() for unwanted_name in unwanted_names):
            objects_to_remove.append(idx)
    
    # Filter out polygons from the frames
    for frame in data["frames"]:
        frame["polygon"] = [
            poly for poly in frame["polygon"] 
            if poly["object"] not in objects_to_remove
        ]
    
    # Filter out objects from the objects list
    data["objects"] = [
        obj for idx, obj in enumerate(data["objects"]) 
        if idx not in objects_to_remove
    ]
    
    return data

# def find_most_prominent_object(data, ws=1.3):
#     object_info = []

#     unwanted_names = ["wall", "wal", "floor", "flor", "floro","ceiling"]

#     # Build a set of unwanted object indices
#     unwanted_indices = set()
#     for idx, obj in enumerate(data["objects"]):
#         # Debug statement
#         # print(f"Processing object at index {idx}: {obj}")

#         # Check if obj is a dict and has 'name' key
#         if isinstance(obj, dict) and "name" in obj:
#             obj_name = obj["name"].lower()
#             if any(unwanted_name in obj_name for unwanted_name in unwanted_names):
#                 unwanted_indices.add(idx)
#         else:
#             # Handle cases where obj is not as expected
#             # print(f"Skipping object at index {idx} due to unexpected structure.")
#             continue  # Skip this object

#     # Iterate over polygons and collect info, skipping unwanted objects
#     for poly in data["frames"][0]["polygon"]:
#         obj_idx = poly["object"]
#         if obj_idx in unwanted_indices:
#             continue  # Skip unwanted objects

#         polygon_points = [(x, y) for x, y in zip(poly["x"], poly["y"])]
#         area = calculate_bounding_box_area(polygon_points)
#         if "XYZ" in poly:
#             average_depth = calculate_average_depth(poly)
#         else:
#             average_depth = float("inf")

#         object_info.append((obj_idx, area, average_depth))

#     if not object_info:
#         print("No valid objects found after filtering unwanted names.")
#         return None  # No objects to consider

#     # Sort objects by area in descending order
#     object_info.sort(key=lambda x: x[1], reverse=True)

#     # Determine the most prominent object
#     if len(object_info) == 1 or object_info[0][1] > ws * object_info[1][1]:
#         # Case 1: The largest object is significantly larger than the second largest
#         most_prominent_object_idx = object_info[0][0]
#     else:
#         # Case 2: Consider both size and depth
#         size_ranking = {
#             obj[0]: i + 1
#             for i, obj in enumerate(
#                 sorted(object_info, key=lambda x: x[1], reverse=True)
#             )
#         }
#         depth_ranking = {
#             obj[0]: i + 1
#             for i, obj in enumerate(sorted(object_info, key=lambda x: x[2]))
#         }
#         combined_ranking = {
#             obj_id: size_ranking[obj_id] + depth_ranking[obj_id]
#             for obj_id, _, _ in object_info
#         }

#         # The object with the lowest combined ranking score is the most prominent
#         most_prominent_object_idx = min(combined_ranking, key=combined_ranking.get)

#     # Get the name of the most prominent object
#     most_prominent_object = data["objects"][most_prominent_object_idx]
#     if isinstance(most_prominent_object, dict) and "name" in most_prominent_object:
#         most_prominent_object_name = most_prominent_object["name"]
#     else:
#         most_prominent_object_name = "Unknown"

#     # print(f"The most prominent object is: {most_prominent_object_name}")
#     return most_prominent_object_name


def read_json(file_path):
    with open(file_path) as f:
        # Load the JSON data
        json_data = json.load(f)

    return json_data


def get_all_paths():
    sund3d_folder = "SUNRGBD/xtion/sun3ddata/"
    other_paths = [
        "SUNRGBD/kv1/b3dodata/",
        "SUNRGBD/kv1/NYUdata/",
        "SUNRGBD/kv2/align_kv2/",
        "SUNRGBD/kv2/kinect2data/",
        "SUNRGBD/realsense/lg/",
        "SUNRGBD/realsense/sa/",
        "SUNRGBD/realsense/sh/",
        "SUNRGBD/realsense/shr/",
        "SUNRGBD/xtion/xtion_align_data/",
    ]

    all_sun3d_paths = []
    sun3ddata_sub_folder_names = os.listdir(sund3d_folder)

    # Iterate through subfolders in sun3ddata
    for sun3d_subfiles in sun3ddata_sub_folder_names:
        sun3d_subfolders = os.listdir(os.path.join(sund3d_folder, sun3d_subfiles))

        # Iterate through subfolders in sun3d_subfiles
        for each_folder in sun3d_subfolders:
            all_sun3d_paths.append(
                os.path.join(sund3d_folder, sun3d_subfiles, each_folder)
            )

    all_paths = other_paths + all_sun3d_paths
    return all_paths


def top_n_frequent_items(lst, n):
    counts = Counter(lst)
    return counts.most_common(n)


def generate_question():
    return "What is the most prominent object?"

def main():
    current_directory = os.getcwd()
    print("Current Directory:", current_directory)
    
    directory_path = "dataset/SUNRGBD_Dataset/"
    os.chdir(directory_path)

    data_split = "validation"
    image_paths = read_paths("splits_output_paths/"+ data_split +"/all_rgb.txt")
    depth_paths = read_paths("splits_output_paths/"+ data_split +"/all_depth.txt")
    annotation_paths = read_paths("splits_output_paths/"+ data_split +"/annotations.txt")

    questions = []
    answers = []
    image_path_final = []
    depth_path_final = []
    item_ids = []

    floor_counter = 0
    wall_counter =-0

    error_counter = 0
    counter = 0
    no_prominent_counter = 0
    data_counter = 1    
    with tqdm(total=len(image_paths)) as pbar:
        for image_path, depth_path, annotation_path in (
            zip(image_paths, depth_paths, annotation_paths)
        ):
            # if image_path == "SUNRGBD/realsense/lg/2014_10_26-13_56_11-1311000073/image/0000074.jpg":
            try:
                with open(annotation_path, "r") as file:
                    annotation_data = json.load(file)

                most_prominent = "Cannot answer"
                #if most_prominent_object is not found then it will be "Cannot answer"
                #however if its found it will be replaced by the object name
                most_prominent = find_most_prominent_object(annotation_data)
                most_prominent = most_prominent.lower()
                most_prominent = process_text_only(most_prominent)

                if not most_prominent or most_prominent == 'n/a' or pd.isna(most_prominent):  # This checks for both empty strings and None
                    most_prominent = "Cannot answer"
                    no_prominent_counter += 1


                if most_prominent == "floor" and floor_counter < 500:
                    questions.append(generate_question())
                    answers.append(most_prominent)
                    image_path_final.append(image_path)
                    depth_path_final.append(depth_path)

                if most_prominent == "wall" and wall_counter < 450:
                    questions.append(generate_question())
                    answers.append(most_prominent)
                    image_path_final.append(image_path)
                    depth_path_final.append(depth_path)

                if most_prominent != "floor" and most_prominent != "wall":
                    questions.append(generate_question())
                    answers.append(most_prominent)
                    image_path_final.append(image_path)
                    depth_path_final.append(depth_path)
                
                item_ids.append(data_counter)
                data_counter += 1
            
                # if(counter == 100):
                #     break
                # counter += 1


            except Exception as e:
                error_counter += 1
                # print(e)
                continue

            
            pbar.set_postfix({"errors": error_counter, "processed": data_counter})
            pbar.update(1)





    df = pd.DataFrame(
        {   
            "IDs": item_ids,
            "Questions": questions,
            "Answers": answers,
            "Image_Path": image_path_final,
            "Depth_Path": depth_path_final,
            "Question_Type": "Object Identification",
        }
    )


    df.to_csv("SUNRGBD/csv_data/individual_datasets/"+ data_split +"/object_identification.csv", index=False)


    print(f"Number of errors: {error_counter}")
    print(f"Number of no prominent object: {no_prominent_counter}")
    print(f"Number of data: {data_counter}")


if __name__ == "__main__":
    main()
