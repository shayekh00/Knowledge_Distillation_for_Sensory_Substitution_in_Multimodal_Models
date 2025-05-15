import json
import numpy as np
import os
import cv2
from  utils import read_paths, create_polygon_points ,find_most_prominent_object,find_object_index
from tqdm import tqdm
import pandas as pd
from PIL import Image
from PIL import Image, ImageDraw, ImageFont
from post_process import process_text_only

# Define the directory containing the file paths
directory_path = "dataset/SUNRGBD_Dataset/"
# Change the current working directory to the specified directory
os.chdir(directory_path)

proximity_questions = []
proximity_answers = []
path_of_the_images = []
path_of_the_depth = []

def find_center_of_mass(polygon_points):
    # Calculate centroid (center of mass)
    centroid_x = sum(point[0] for point in polygon_points) / len(polygon_points)
    centroid_y = sum(point[1] for point in polygon_points) / len(polygon_points)

    centroid = (centroid_x, centroid_y)
    return centroid

def normalize_coordinates(coord, max_val):
    return coord / max_val

def compute_distance(coord_a, coord_b):
    return np.sqrt((coord_a[0] - coord_b[0])**2 + (coord_a[1] - coord_b[1])**2)

def find_closest_object(object_with_centroid,current_object):
    closest_object = None
    min_distance = float('inf')
    current_centroid = object_with_centroid[current_object]
    

    for object_label, centroid in object_with_centroid.items():
        if object_label != current_object:
            distance = compute_distance(centroid, current_centroid)
            
            if distance < min_distance:  # Ensure we're comparing scalar values
                min_distance = distance
                closest_object = object_label

    return closest_object

def extract_closest_object(annotation_data, image_path, depth_path):
    object_with_centroid = {}
    all_objects = annotation_data["objects"]

    # Extracting object centroids
    polygon = annotation_data["frames"][0]["polygon"]
    unwanted_names = ["wall", "wal", "floor", "flor", "floro", "ceiling"]
    for obj in polygon:
        object_label = obj['object']
        label_name = all_objects[object_label]["name"]

        if any(unwanted_name in label_name.lower() for unwanted_name in unwanted_names):
            continue
        
        x = obj['x']
        y = obj['y']
        polygon_points = [(x, y) for x, y in zip(x, y)]
        centroid = find_center_of_mass(polygon_points)
        object_with_centroid[object_label] = centroid

    # Load image and depth
    image = cv2.imread(image_path)
    # depth = cv2.imread(depth_path)

    # For each object, find the closest object
    closest_objects = {}
    for object_label in object_with_centroid:
        current_object = object_label
        closest_objects[object_label] = find_closest_object(object_with_centroid,current_object)

    return closest_objects,image,object_with_centroid



def find_farthest_object(object_with_centroid, current_object):
    farthest_object = None
    max_distance = -float('inf')  # Initialize to negative infinity
    current_centroid = object_with_centroid[current_object]

    for object_label, centroid in object_with_centroid.items():
        if object_label != current_object:
            distance = compute_distance(centroid, current_centroid)
            
            if distance > max_distance:  # Adjusted comparison
                max_distance = distance
                farthest_object = object_label

    return farthest_object

def extract_farthest_object(annotation_data, image_path, depth_path):
    object_with_centroid = {}

    # Extracting object centroids
    polygon = annotation_data["frames"][0]["polygon"]
    for obj in polygon:
        object_label = obj['object']
        x = obj['x']
        y = obj['y']
        polygon_points = [(x, y) for x, y in zip(x, y)]
        centroid = find_center_of_mass(polygon_points)
        object_with_centroid[object_label] = centroid

    # Load image and depth
    image = cv2.imread(image_path)
    # depth = cv2.imread(depth_path)

    # For each object, find the farthest object
    farthest_objects = {}
    for object_label in object_with_centroid:
        current_object = object_label
        farthest_objects[object_label] = find_farthest_object(object_with_centroid, current_object)

    return farthest_objects, image, object_with_centroid

def extract_farthest_object_from_one_object(annotation_data, image_path, most_prominent_object_index):
    object_with_centroid = {}

    # Extracting object centroids
    polygon = annotation_data["frames"][0]["polygon"]
    for obj in polygon:
        object_label = obj['object']
        x = obj['x']
        y = obj['y']
        polygon_points = [(x, y) for x, y in zip(x, y)]
        centroid = find_center_of_mass(polygon_points)
        object_with_centroid[object_label] = centroid


    farthest_object = find_farthest_object(object_with_centroid, most_prominent_object_index)

    return farthest_object


def get_object_labels(annotated_data):
    object_id_set = []  # Use a set to store unique object labels
    for frame in annotated_data['frames']:
        for polygon in frame['polygon']:
            object_id = polygon['object']
            object_id_set.append(object_id)
            
    return object_id_set


def get_object_name(annotated_data,index):
    object_id_names = []
    object_name = annotated_data['objects'][index]['name']
    return object_name

def map_lists_to_dict(keys, values):
    result_dict = {}
    for key, value in zip(keys, values):
        result_dict[key] = value
    return result_dict

# Function to draw centroids on the image
def draw_centroids(image, centroids):
    for label, centroid in centroids.items():
        x, y = centroid
        cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)
        cv2.putText(image, str(label), (int(x) - 10, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Display the image
    # cv2.imshow("Image with Centroids", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# Function to draw only the centroids for closest and farthest objects
def draw_centroids_pil(image, centroids, closest_object=None, farthest_object=None):
    # Convert OpenCV image to RGB (PIL uses RGB format)
    opencv_image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Create a PIL Image from the RGB array
    image = Image.fromarray(opencv_image_rgb)

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    # Only draw the centroids for the closest and farthest objects
    for label, centroid in centroids.items():
        if label == closest_object or label == farthest_object:
            x, y = centroid
            color = (255, 0, 0)  # Red for closest and farthest objects
            # Draw the centroid
            draw.ellipse([(x-5, y-5), (x+5, y+5)], fill=color)
            draw.text((x-10, y-10), str(label), fill=(0, 0, 0), font=font)

    # Save the image with centroids drawn on it
    image.save('output_image_with_selected_centroids.jpg')


def get_object_name_by_labels(annotation_data):
    object_id_to_name = {}
    object_ids_list = get_object_labels(annotation_data)
    for index in object_ids_list:
        object_id_to_name[index] = get_object_name(annotation_data, index)
    return object_id_to_name

def convert_object_ids_to_names(closest_objects, object_id_to_name):
    closest_objects_names = {}
    unique_pairs = set()
    for key, value in closest_objects.items():
        pair = (object_id_to_name[key], object_id_to_name[value])
        if pair not in unique_pairs and (pair[1], pair[0]) not in unique_pairs:
            closest_objects_names[object_id_to_name[key]] = object_id_to_name[value]
            unique_pairs.add(pair)
    return list(unique_pairs)



def generate_proximity_question_closest(most_prominent_object_name , closest_object_name, image_path , depth_path):
    
    question = f"What is closest to the {most_prominent_object_name}?"
    answer = closest_object_name
    proximity_questions.append(question)
    proximity_answers.append(answer)
    path_of_the_images.append(image_path)
    path_of_the_depth.append(depth_path)

def generate_proximity_question_farthest(most_prominent_object_name , closest_object_name, image_path , depth_path):
    
    question = f"What is farthest to the {most_prominent_object_name}?"
    answer = closest_object_name
    proximity_questions.append(question)
    proximity_answers.append(answer)
    path_of_the_images.append(image_path)
    path_of_the_depth.append(depth_path)  

def get_keys_from_value(d, value):
    return [key for key, val in d.items() if val == value]



def extract_closest_object_from_one_object(annotation_data, image_path ,most_prominent_object_index):
    object_with_centroid = {}
    all_objects = annotation_data["objects"]
    # Extracting object centroids
    polygon = annotation_data["frames"][0]["polygon"]
    unwanted_names = ["wall", "wal", "floor", "flor", "floro", "ceiling"]

    for obj in polygon:
        object_label = obj['object']

        label_name = all_objects[object_label]["name"]

        if any(unwanted_name in label_name.lower() for unwanted_name in unwanted_names):
            continue

        x = obj['x']
        y = obj['y']
        polygon_points = [(x, y) for x, y in zip(x, y)]
        centroid = find_center_of_mass(polygon_points)
        object_with_centroid[object_label] = centroid


    closest_object = find_closest_object(object_with_centroid,most_prominent_object_index)


    return closest_object

def main():
    segmentation_label_paths = read_paths('all_segmentation_labels.txt')

    # image_paths = read_paths('all_rgb.txt')
    # depth_paths = read_paths('all_depth.txt')
    # annotation_paths = read_paths('annotations.txt')


    data_split = "train"

    image_paths = read_paths("splits_output_paths/"+ data_split +"/all_rgb.txt")
    depth_paths = read_paths("splits_output_paths/"+ data_split +"/all_depth.txt")
    annotation_paths = read_paths("splits_output_paths/"+ data_split +"/annotations.txt")


    data_counter = 1
    error_counter = 0
    item_ids = []
    with tqdm(total=len(image_paths)) as pbar:
        for image_path, depth_path, annotation_path in tqdm(zip(image_paths, depth_paths, annotation_paths)):
            try:
                object_id_to_name = {}

                with open(annotation_path, 'r') as file:
                    annotation_data = json.load(file)

                most_prominent_object_name = find_most_prominent_object(annotation_data)
                most_prominent_object_index = find_object_index(annotation_data, most_prominent_object_name)
                # print("most_prominent_object_name: ", most_prominent_object_name)

                
                # Get closest and farthest objects
                closest_object = extract_closest_object_from_one_object(annotation_data, image_path, most_prominent_object_index)
                # farthest_object = extract_farthest_object_from_one_object(annotation_data, image_path, most_prominent_object_index)
                
                object_id_to_name = get_object_name_by_labels(annotation_data)

                # Map object IDs to names for closest and farthest objects
                closest_object_name = object_id_to_name.get(closest_object, None)
                # farthest_object_name = object_id_to_name.get(farthest_object, None)

                most_prominent_object_name = process_text_only(most_prominent_object_name)
                # print("closest_object_name b4: ", closest_object_name)
                closest_object_name = process_text_only(closest_object_name)

                # print("closest_object_name: ", closest_object_name)
                # print("most_prominent_object_name: ", most_prominent_object_name)
                # farthest_object_name = process_text_only(farthest_object_name)


                # Generate proximity questions
                generate_proximity_question_closest(most_prominent_object_name, closest_object_name, image_path, depth_path)

                # generate_proximity_question_farthest(most_prominent_object_name, farthest_object_name, image_path, depth_path)

                # # Extract centroids for visualization
                # _, image, object_with_centroid = extract_closest_object(annotation_data, image_path, depth_path)
                # draw_centroids_pil(image, object_with_centroid, closest_object, farthest_object)  # Draw centroids with special colors

                # if(data_counter == 6):
                #     break

                item_ids.append(data_counter)
                

            except Exception as e:
                error_counter += 1
                continue

            data_counter += 1
            pbar.set_postfix({"errors": error_counter, "processed": data_counter})
            pbar.update(1)



    # Create DataFrame for questions and answers
    df = pd.DataFrame({
        "IDs": item_ids,
        'Questions': proximity_questions,
        'Answers': proximity_answers,
        'Image_Path': path_of_the_images,
        'Depth_Path': path_of_the_depth,
        "Question_Type": "Proximity"
    })

    df.to_csv("SUNRGBD/csv_data/individual_datasets/"+ data_split +"/proximity_qa.csv", index=False)
    # df.to_csv("SUNRGBD/csv_data/individual_datasets/new/proximity_qa.csv", index=False)
    print(f"Number of errors: {error_counter}")
    print(f"Number of data total: {data_counter}")

                



if __name__ == "__main__":
    main()


