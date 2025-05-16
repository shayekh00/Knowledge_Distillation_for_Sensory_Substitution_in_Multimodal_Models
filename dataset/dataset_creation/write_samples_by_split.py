import json
import numpy as np
import os
import cv2
from tqdm import tqdm
import pandas as pd
from utils import read_paths
import inflect
from PIL import Image

import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
current_path = os.getcwd()
print("Current working directory:", current_path)

directory_path = "dataset/SUNRGBD_Dataset/"
# Change the current working directory to the specified directory
os.chdir(directory_path)

current_path = os.getcwd()
print("Current working directory:", current_path)


def main():
    data_split = "test"
    
    df = pd.read_csv("SUNRGBD/csv_data/individual_datasets/"+data_split+"_dataset.csv")
    image_paths = df["Image_Path"].tolist()
    question_ids = df["Question_Id"].tolist()


    data_counter = 1
    
    p = inflect.engine()

    with tqdm(total=len(image_paths)) as pbar:
        for image_path, question_id in (
            zip(image_paths, question_ids)
        ):
            output_path = f'sample_check/' + data_split + '_set_final/' +  str(question_id) + '.jpg'
            image_to_outpput = np.array( Image.open(image_path).convert('RGB') )
            cv2.imwrite(output_path, image_to_outpput)
            # print(f"Image saved to {output_path}")
            data_counter += 1
            
            pbar.set_postfix({ "processed": data_counter})
            pbar.update(1)



if __name__ == "__main__":
    main()
