import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import sys

from inference_utils import (
    extract_val_loss,
    remove_substring_from_path,
    get_image_paths,
    convert_numbers_to_words,
    convert_depth_image_into_3D,
    get_model_answer,
    load_environment,
    initialize_model,
    get_model_answer
)


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
print("sys.path:", sys.path)
current_dir = os.getcwd()
print("Current directory:", current_dir)




# if __name__ == "__main__":
DATA_POINT_INDEX = 0  # Index of the data point to process
DATA_SPLIT = "val"    # or "test"
checkpoint_filename = "llava_onevision_checkpoint_double_trouble_phase2_-epoch=00-val_loss=0.0076.ckpt" 


phase_no = 2
phase = "phase"+str(phase_no)
phase_param = phase_no
kd_model_type = "double_trouble"
pixel_data_type = "depth"


# Load environment variables and initialize the model
ROOT_DATA_DIR, MAIN_ROOT_DIR = load_environment()

processor, model, pad_token_id = initialize_model(MAIN_ROOT_DIR, model_id="llava-hf/llava-onevision-qwen2-0.5b-ov-hf", 
                                                    load_checkpoint=False, model_type=kd_model_type, phase_param=phase_param,
                                                    checkpoint_filename=checkpoint_filename) 
                                                    

# Paths to CSV files

csv_file_path_val = os.path.join(ROOT_DATA_DIR, "SUNRGBD/csv_data", "val_dataset.csv")
csv_file_path_test = os.path.join(ROOT_DATA_DIR, "SUNRGBD/csv_data", "test_dataset.csv")

# Read the CSV files
df_val = pd.read_csv(csv_file_path_val)
df_test = pd.read_csv(csv_file_path_test)



df = df_val if DATA_SPLIT == "val" else df_test


results = []
predictions = []
references = []

results_df = pd.DataFrame(columns=['Question_Id', 'Questions', 'Question_Type', 'Answers',"Model_Answer"])
data_list = []

print("Image used :", pixel_data_type)


row = df.iloc[DATA_POINT_INDEX]
question_id = int(row["Question_Id"])
question = row["Questions"]
question_type = df.iloc[DATA_POINT_INDEX]['Question_Type']
answers = row["Answers"].strip().lower()

rgb_image_path, depth_image_path = get_image_paths(
    row["Image_Path"], row["Depth_Path"], ROOT_DATA_DIR
)

rgb_image = Image.open(rgb_image_path)
rgb_image_np = np.array(rgb_image)

depth_image_np = convert_depth_image_into_3D(depth_image_path )


image_used = depth_image_np if pixel_data_type == "depth" else rgb_image_np



model_final_answer = get_model_answer(
    question, image_used, processor, model, pad_token_id
)


data_list.append({
    'Question_Id': question_id,
    'Questions': question,
    'Question_Type': question_type,
    'Answers': answers,
    'Model_Answer': model_final_answer
})

predictions.append(model_final_answer)
references.append(answers)


#Write the results to a CSV file
results_df = pd.DataFrame(data_list)
print("Results DataFrame:")
print(results_df)