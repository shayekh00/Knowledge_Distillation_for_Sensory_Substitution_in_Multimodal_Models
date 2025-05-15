import json
import numpy as np
from skimage.color import rgb2lab, deltaE_cie76
from PIL import Image
import os
import cv2
from PIL import Image, ImageDraw
from tqdm import tqdm
from utils import read_paths
import color_extraction
import pandas as pd
from utils import read_paths ,find_most_prominent_object
from post_process import process_text_only
import torch
# Define the directory containing the file paths
directory_path = "dataset/SUNRGBD_Dataset/"
# Change the current working directory to the specified directory
os.chdir(directory_path)

# # Paths to your files - these need to be updated according to your dataset location
# annotations_path = "annotations.txt"
# image_paths_file = "all_rgb.txt"
# segmentation_labels_file = "all_segmentation_labels.txt"


def load_annotations(annotations_path):
    with open(annotations_path, "r") as file:
        return [line.strip() for line in file.readlines()]


def read_image_and_convert(image_path):
    image = Image.open(image_path)
    image = np.array(image)
    return rgb2lab(image)


# def read_paths(file_path):
#     with open(file_path, 'r') as file:
#         paths = [line.strip() for line in file.readlines()]
#     return paths


def crop_polygon(image_path, polygon):
    # Load the original image
    original_image = Image.open(image_path)
    original_image = original_image.convert("RGBA")
    width, height = original_image.size

    # Create a blank mask image to draw the polygon
    mask_image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_image)
    draw.polygon(polygon, outline=1, fill=1)

    # Create a new image for the cropped result
    # cropped_image = Image.new("RGBA", original_image.size)
    cropped_image = Image.new("RGB", original_image.size)

    # Iterate over each pixel in the original image
    for x in range(width):
        for y in range(height):
            # Check if the pixel is within the polygon
            if mask_image.getpixel((x, y)):
                # Copy the pixel onto the cropped image
                cropped_image.putpixel((x, y), original_image.getpixel((x, y)))

    return cropped_image


def pil_to_opencv(pil_image):
    # Convert the PIL image to a NumPy array
    numpy_image = np.array(pil_image)

    # Convert the RGB image to BGR (OpenCV uses BGR format)
    opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

    return opencv_image


def crop_black_parts(img):

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold the image to create a binary image where black regions are 0 and non-black are 255
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Find contours from the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the bounding rectangle for all contours
    x, y, w, h = cv2.boundingRect(np.vstack(contours))

    # Crop the original image with the bounding rectangle dimensions
    cropped_img = img[y : y + h, x : x + w]

    return cropped_img


def get_color_name(cropped_image):
    colors = color_extraction.get_counts(cropped_image)
    # Sort the dictionary items by values in descending order
    sorted_colors = sorted(colors.items(), key=lambda x: x[1], reverse=True)
    # Extract the second highest color
    second_highest_color = sorted_colors[1][0]

    return second_highest_color


def generate_object_question(object_name):
    return f"What is the color of the {object_name}?"

# from transformers import ViltProcessor, ViltForQuestionAnswering
# from PIL import Image
# import torch
# # Initialize the model and processor once
# processor = ViltProcessor.from_pretrained('dandelin/vilt-b32-finetuned-vqa')
# model = ViltForQuestionAnswering.from_pretrained('dandelin/vilt-b32-finetuned-vqa')

# def get_main_object_color(image):
#     """
#     Determines the color of the main object in an image.

#     Parameters:
#     - image (str or PIL.Image.Image): The image file path or PIL Image object.

#     Returns:
#     - str: The color of the main object as determined by the model.
#     """
#     # Load the image if a file path is provided
#     if isinstance(image, str):
#         image = Image.open(image)
#     elif not isinstance(image, Image.Image):
#         raise ValueError("The 'image' parameter must be a file path or a PIL Image object.")

#     # Define the question
#     question = "What is the dominant color of the most prominent object?"

#     # Prepare inputs for the model
#     inputs = processor(image, question, return_tensors="pt")

#     # Get the model's prediction
#     with torch.no_grad():
#         outputs = model(**inputs)
#     logits = outputs.logits

#     # Decode the answer
#     answer = model.config.id2label[logits.argmax(-1).item()]

#     return answer

from transformers import BlipProcessor, BlipForQuestionAnswering

# Load the model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

def get_main_object_color_blip(image,most_prominent):
    """
    Determines the color of the main object in an image using BLIP.

    Parameters:
    - image (str or PIL.Image.Image): The image file path or PIL Image object.

    Returns:
    - str: The color of the main object as determined by the model.
    """
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    elif isinstance(image, Image.Image):
        image = image.convert('RGB')
    else:
        raise ValueError("The 'image' parameter must be a file path or a PIL Image object.")

    question = "What is the color of "+ most_prominent + "?"
    inputs = processor(images=image, text=question, return_tensors="pt")

    with torch.no_grad():
        generated_ids = model.generate(**inputs)
    answer = processor.decode(generated_ids[0], skip_special_tokens=True)

    return answer.strip()



def main():

    # image_paths = read_paths("all_rgb.txt")
    segmentation_label_paths = read_paths("all_segmentation_labels.txt")
    # annotation_paths = read_paths("annotations.txt")
    # depth_paths = read_paths("all_depth.txt")

    data_split = "validation"
    image_paths = read_paths("splits_output_paths/"+ data_split +"/all_rgb.txt")
    depth_paths = read_paths("splits_output_paths/"+ data_split +"/all_depth.txt")
    annotation_paths = read_paths("splits_output_paths/"+ data_split +"/annotations.txt")

    questions = []
    answers = []
    image_path_final = []
    depth_path_final = []

    error_counter = 0
    data_counter = 1


    item_ids = []
    with tqdm(total=len(image_paths)) as pbar:
        for image_path, segmentation_path, annotation_path, depth_path in (
            zip(image_paths, segmentation_label_paths, annotation_paths, depth_paths) ):
            try:

                with open(annotation_path, "r") as file:
                    annotation_data = json.load(file)

                most_prominent_object_name = find_most_prominent_object(annotation_data)
                most_prominent = process_text_only(most_prominent_object_name)

                # polygon = annotation_data["frames"][0]["polygon"]
                # all_objects = annotation_data["objects"]

                color_from_ai = get_main_object_color_blip(image_path, most_prominent)

                questions.append(generate_object_question(most_prominent))
                answers.append(color_from_ai)
                image_path_final.append(image_path)
                depth_path_final.append(depth_path)
                item_ids.append(data_counter)
                
                
                #uncomment this three lines to save the images
                # output_path = f'sample_check/color_check_test_set/full_image_{data_counter}_{color_from_ai}_{most_prominent}.jpg'
                # image_to_outpput = np.array( Image.open(image_path).convert('RGB') )
                # cv2.imwrite(output_path, image_to_outpput)



            except Exception as e:
                error_counter += 1
                # print(e)
                continue

            data_counter += 1
            pbar.set_postfix({"errors": error_counter, "processed": data_counter})
            pbar.update(1)

        # break
    # Create DataFrame
    df = pd.DataFrame(
        {   
            "IDs": item_ids,
            "Questions": questions,
            "Answers": answers,
            "Image_Path": image_path_final,
            "Depth_Path": depth_path_final,
            "Question_Type" : "Color Identification"
        }
    )

    df.to_csv("SUNRGBD/csv_data/individual_datasets/"+ data_split +"/color_questions.csv", index=False)

    print(f"Number of errors: {error_counter}")
    print(f"Number of data total: {data_counter}")


if __name__ == "__main__":
    main()
