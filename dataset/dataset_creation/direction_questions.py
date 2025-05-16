import numpy as np
from shapely.geometry import Polygon
import os
import json
from PIL import Image
from tqdm import tqdm
from collections import Counter
import pandas as pd
from utils import read_paths ,find_most_prominent_object ,find_object_index ,find_object_polygon
import math
from post_process import process_text_only
# ws value
ws = 1.3




def read_json(file_path):
    with open(file_path) as f:
        # Load the JSON data
        json_data = json.load(f)

    return json_data

def find_center_of_mass(polygon_points):
    centroid_x = sum(point[0] for point in polygon_points) / len(polygon_points)
    centroid_y = sum(point[1] for point in polygon_points) / len(polygon_points)
    return centroid_x, centroid_y

def get_relative_position(angle):
    if -math.pi/2 < angle <= math.pi/2:
        return "left"
    else:
        return "right"

def get_vertical_position(angle):
    if 0 < angle <= math.pi:
        return "above"
    else:
        return "under"

def generate_qa_from_directions(directions):
    qas = []
    for label1, label2, direction_vector, angle in directions:
        horizontal_pos = get_relative_position(angle)
        vertical_pos = get_vertical_position(angle)
        
        # Generating a more nuanced position description
        if abs(angle) < math.pi/4 or abs(angle) > 3*math.pi/4:
            position = horizontal_pos
        else:
            position = vertical_pos
        
        question = f"Where is {label1}?"
        answer = f"{label1} is {position} of {label2}."
        qas.append((question, answer))
    return qas


# def generate_single_direction_question(directions ,most_prominent_object_processed):
#     # Take the first direction from the list
#     label1, label2, direction_vector, angle = directions[0]

#     label2_processed = process_text_only(label2)
    
#     horizontal_pos = get_relative_position(angle)
#     vertical_pos = get_vertical_position(angle)
    
#     # Generating a more nuanced position description
#     if abs(angle) < math.pi/4 or abs(angle) > 3*math.pi/4:
#         position = horizontal_pos
#     else:
#         position = vertical_pos
    
#     question = f"Where is {most_prominent_object_processed} relative to {label2_processed}?"
#     # answer = f"{label1} is {position} of {label2}."
#     answer = position
    
#     return question, answer

def generate_single_direction_question(directions, most_prominent_object_processed):
    # Take the first direction from the list
    label1, label2, direction_vector, angle = directions[0]

    label2_processed = process_text_only(label2)

    # Determine horizontal and vertical positions
    horizontal_pos = get_relative_position(angle)
    vertical_pos = get_vertical_position(angle)

    # Determine the position based on the angle
    if math.isclose(angle, 0, abs_tol=1e-2):  # Exactly right
        position = "right"
    elif math.isclose(angle, math.pi, abs_tol=1e-2) or math.isclose(angle, -math.pi, abs_tol=1e-2):  # Exactly left
        position = "left"
    elif math.isclose(angle, math.pi / 2, abs_tol=1e-2):  # Exactly above
        position = "above"
    elif math.isclose(angle, -math.pi / 2, abs_tol=1e-2):  # Exactly under
        position = "under"
    else:  # Combination of horizontal and vertical
        position = f"{horizontal_pos} and {vertical_pos}"
    
    # Generate the question and answer
    question = f"Where is {most_prominent_object_processed} relative to {label2_processed}?"
    answer = position
    
    return question, answer

def calculate_directions_between_objects(centroids):
    directions = []
    for i, (centroid1, label1) in enumerate(centroids):
        for j, (centroid2, label2) in enumerate(centroids):
            if i != j:  # Ensuring not to calculate direction to itself
                direction_vector = (centroid2[0] - centroid1[0], centroid2[1] - centroid1[1])
                angle = math.atan2(direction_vector[1], direction_vector[0])
                directions.append((label1, label2, direction_vector, angle))
    return directions

def calculate_directions_between_objects_and_most_prominent(centroids, most_prominent_object):
    directions = []
    prominent_centroid = None

    # Find the centroid of the most prominent object
    for centroid, label in centroids:
        if label == most_prominent_object:
            prominent_centroid = centroid
            break
    
    if prominent_centroid is None:
        raise ValueError("Most prominent object not found in the centroids list")

    for centroid, label in centroids:
        if label != most_prominent_object:
            direction_vector = (centroid[0] - prominent_centroid[0], centroid[1] - prominent_centroid[1])
            angle = math.atan2(direction_vector[1], direction_vector[0])
            directions.append((most_prominent_object, label, direction_vector, angle))
    
    return directions


def normalize_polygon_coordinates(obj):
    """
    Normalizes the 'x' and 'y' fields of a polygon object to ensure they are processable.
    """
    # Ensure 'x' and 'y' are lists
    if not isinstance(obj["x"], list):
        obj["x"] = [obj["x"]] if isinstance(obj["x"], (int, float)) else []
    if not isinstance(obj["y"], list):
        obj["y"] = [obj["y"]] if isinstance(obj["y"], (int, float)) else []

    # Ensure 'x' and 'y' have the same length
    min_length = min(len(obj["x"]), len(obj["y"]))
    if min_length == 0:
        # If one of them is empty, return an empty polygon
        return []

    obj["x"] = obj["x"][:min_length]
    obj["y"] = obj["y"][:min_length]

    # Return the normalized polygon points
    return [(x, y) for x, y in zip(obj["x"], obj["y"])]



def main():
    current_directory = os.getcwd()
    print("Current Directory:", current_directory)
    
    directory_path = "dataset/SUNRGBD_Dataset/"
    os.chdir(directory_path)

    data_split = "test"

    image_paths = read_paths("splits_output_paths/"+ data_split +"/all_rgb.txt")
    depth_paths = read_paths("splits_output_paths/"+ data_split +"/all_depth.txt")
    annotation_paths = read_paths("splits_output_paths/"+ data_split +"/annotations.txt")

    questions = []
    answers = []
    image_path_final = []
    depth_path_final = []

    item_ids = []

    data_counter = 1
    error_counter = 0
    unwanted_names = ["wall", "wal", "floor", "flor", "floro", "ceiling"]

    with tqdm(total=len(image_paths)) as pbar:
        for image_path, depth_path, annotation_path in (
            zip(image_paths, depth_paths, annotation_paths)
        ):
            try:
                with open(annotation_path, "r") as file:
                    annotation_data = json.load(file)


                most_prominent_object = find_most_prominent_object(annotation_data)
                most_prominent_object_processed = process_text_only(most_prominent_object)

                # most_prominent_object_index = find_object_index(annotation_data, most_prominent_object)
                # most_prominent_object_polygon = find_object_polygon(annotation_data, most_prominent_object_index)
                # print(most_prominent_object)
                
                polygon = annotation_data["frames"][0]["polygon"]
                all_objects = annotation_data["objects"]

                # break
                centroids = []

                # for obj in polygon:
                #     object_label = obj["object"]
                #     label_name = all_objects[object_label]["name"]


                #     polygon_points = [(x, y) for x, y in zip(obj["x"], obj["y"])]

                #     centroid = find_center_of_mass(polygon_points)
                #     centroids.append((centroid, label_name))
                
                for obj in polygon:
                    object_label = obj["object"]
                    label_name = all_objects[object_label]["name"]
                    if any(unwanted_name in label_name.lower() for unwanted_name in unwanted_names):
                        continue

                    # Normalize the polygon coordinates
                    polygon_points = normalize_polygon_coordinates(obj)

                    # Skip if the polygon is empty after normalization
                    if not polygon_points:
                        # print(f"Skipping invalid or empty polygon for object: {label_name}")
                        continue

                    # Calculate the centroid
                    try:
                        centroid = find_center_of_mass(polygon_points)
                        centroids.append((centroid, label_name))
                        # print("most_prominent_object: ", most_prominent_object)
                        # print("label_name: ", label_name)
                    except Exception as e:
                        # print(f"Error calculating centroid for {label_name}: {e}")
                        continue            

                # Calculate direction from each object to every other
                # directions = calculate_directions_between_objects(centroids)


                directions = calculate_directions_between_objects_and_most_prominent(centroids, most_prominent_object)
                question, answer = generate_single_direction_question(directions, most_prominent_object_processed)
                # qas = generate_single_direction_question(directions)
                

                # image = Image.open(image_path)
                # output_directory = "sample_check/test_set_new"
                # os.makedirs(output_directory, exist_ok=True)  # Create directory if it doesn't exist
                # output_image_path = os.path.join(output_directory, str(data_counter)+".jpg" )
                # image.save(output_image_path)


                
                # print(question)
                # print(answer)
                questions.append(question)
                answers.append(answer)
                image_path_final.append(image_path)
                depth_path_final.append(depth_path)
                item_ids.append(data_counter)
                
                
                        
            except Exception as e:
                error_counter += 1
                # print(e)
                # print("Error in data counter: ", data_counter)
                continue
            
            
            data_counter += 1

            pbar.set_postfix({"errors": error_counter, "processed": data_counter})
            pbar.update(1)


    print(f"Number of errors: {error_counter}")
    print(f"Number of data total: {data_counter}")



    df = pd.DataFrame(
        {   "IDs": item_ids,
            "Questions": questions,
            "Answers": answers,
            "Image_Path": image_path_final,
            "Depth_Path": depth_path_final,
            "Question_Type": "Direction"
        }
    )

    df.to_csv("SUNRGBD/csv_data/individual_datasets/"+ data_split +"/direction_questions.csv", index=False)


if __name__ == "__main__":
    main()
