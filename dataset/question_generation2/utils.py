
from shapely.geometry import Polygon
import numpy as np

print("utils.py from ques gen is being imported")
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


def read_paths(file_path):
    with open(file_path, "r") as file:
        paths = [line.strip() for line in file.readlines()]
    return paths


def create_polygon_points(x, y):
    # Check if x and y are single integers
    if isinstance(x, int) and isinstance(y, int):
        polygon_points = [(x, y)]
    else:
        # Convert single integers to lists
        if not isinstance(x, list):
            x = [x]
        if not isinstance(y, list):
            y = [y]
        # Create polygon points
        polygon_points = [(xi, yi) for xi, yi in zip(x, y)]

    return polygon_points


def clean_and_dedupe(values):
    seen = set()  # Track seen values in their lowercase form
    cleaned = []  # Prepare a list to store the cleaned (deduplicated) list

    for value in values:
        if not isinstance(value, str):
            # Convert non-string values to strings
            value = str(value)

        lowercase_value = value.lower()
        if lowercase_value in seen:
            # If we've seen this value (in lowercase) before, skip adding it again
            continue
        else:
            # Add the value to the cleaned list and mark it as seen
            cleaned.append(value)
            seen.add(lowercase_value)

    return seen


def get_object_name_list(annotation_data):
    object_names = []
    for obj in annotation_data["objects"]:
        try:
            object_names.append(obj["name"])
        except Exception as e:
            continue

    return object_names


def is_number(n):
    return isinstance(n, (int, float, complex))

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

def find_most_prominent_object(data, ws=1.3):
    object_info = []

    unwanted_names = ["wall", "wal", "floor", "flor", "floro","ceiling"]

    # Build a set of unwanted object indices
    unwanted_indices = set()
    for idx, obj in enumerate(data["objects"]):
        # Debug statement
        # print(f"Processing object at index {idx}: {obj}")

        # Check if obj is a dict and has 'name' key
        if isinstance(obj, dict) and "name" in obj:
            obj_name = obj["name"].lower()
            if any(unwanted_name in obj_name for unwanted_name in unwanted_names):
                unwanted_indices.add(idx)
        else:
            # Handle cases where obj is not as expected
            # print(f"Skipping object at index {idx} due to unexpected structure.")
            continue  # Skip this object

    # Iterate over polygons and collect info, skipping unwanted objects
    for poly in data["frames"][0]["polygon"]:
        obj_idx = poly["object"]
        if obj_idx in unwanted_indices:
            continue  # Skip unwanted objects

        polygon_points = [(x, y) for x, y in zip(poly["x"], poly["y"])]
        area = calculate_bounding_box_area(polygon_points)
        if "XYZ" in poly:
            average_depth = calculate_average_depth(poly)
        else:
            average_depth = float("inf")

        object_info.append((obj_idx, area, average_depth))

    if not object_info:
        print("No valid objects found after filtering unwanted names.")
        return None  # No objects to consider

    # Sort objects by area in descending order
    object_info.sort(key=lambda x: x[1], reverse=True)

    # Determine the most prominent object
    if len(object_info) == 1 or object_info[0][1] > ws * object_info[1][1]:
        # Case 1: The largest object is significantly larger than the second largest
        most_prominent_object_idx = object_info[0][0]
    else:
        # Case 2: Consider both size and depth
        size_ranking = {
            obj[0]: i + 1
            for i, obj in enumerate(
                sorted(object_info, key=lambda x: x[1], reverse=True)
            )
        }
        depth_ranking = {
            obj[0]: i + 1
            for i, obj in enumerate(sorted(object_info, key=lambda x: x[2]))
        }
        combined_ranking = {
            obj_id: size_ranking[obj_id] + depth_ranking[obj_id]
            for obj_id, _, _ in object_info
        }

        # The object with the lowest combined ranking score is the most prominent
        most_prominent_object_idx = min(combined_ranking, key=combined_ranking.get)

    # Get the name of the most prominent object
    most_prominent_object = data["objects"][most_prominent_object_idx]
    if isinstance(most_prominent_object, dict) and "name" in most_prominent_object:
        most_prominent_object_name = most_prominent_object["name"]
    else:
        most_prominent_object_name = "Unknown"

    # print(f"The most prominent object is: {most_prominent_object_name}")
    return most_prominent_object_name




# def find_most_prominent_object(data, ws=1.3):
#     object_info = []

#     for obj in data["frames"][0]["polygon"]:
#         polygon_points = [(x, y) for x, y in zip(obj["x"], obj["y"])]
#         area = calculate_bounding_box_area(polygon_points)
#         if "XYZ" in obj:
#             average_depth = calculate_average_depth(obj)
#         else:
#             average_depth = float("inf")

#         object_info.append((obj["object"], area, average_depth))

#     # Sort objects by area in descending order
#     object_info.sort(key=lambda x: x[1], reverse=True)

#     # Determine the most prominent object
#     if object_info[0][1] > ws * object_info[1][1]:
#         # Case 1: The largest object is significantly larger than the second largest
#         most_prominent_object = object_info[0][0]
#     else:
#         # Case 2: Consider both size and depth
#         # Here we use a simple ranking by summing the inverse rankings of size and depth
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
#             obj[0]: size_ranking[obj[0]] + depth_ranking[obj[0]] for obj in object_info
#         }

#         # The object with the lowest combined ranking score is the most prominent
#         most_prominent_object = min(combined_ranking, key=combined_ranking.get)

#     # Get the name of the most prominent object
#     most_prominent_object_name = data["objects"][most_prominent_object]["name"]

#     # print("combined_ranking: ", combined_ranking)

#     return most_prominent_object_name


def find_object_index(annotation_data, most_prominent_object):
    objects = annotation_data["objects"]
    # print("most_prominent_object: ", most_prominent_object) 
    # print("objects: ", objects)

    for i, obj in enumerate(objects):
        obj_name = get_name(obj)
        if obj_name == most_prominent_object:
            return i

    return -1

def get_name(obj):
    if isinstance(obj, dict):
        return obj.get('name', 'Key not found')
    elif isinstance(obj, list):
        return 'Cannot access key in a list'
    else:
        return 'Unsupported type'
    

def find_object_polygon(annotation_data, most_prominent_object_index):
    polygons = annotation_data["frames"][0]["polygon"]

    for polygon in polygons:
        if polygon["object"] == most_prominent_object_index:
            return polygon
        
    return None


