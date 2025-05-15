import torch
from transformers import AutoProcessor
import pandas as pd
from PIL import Image
from tqdm import tqdm
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
current_dir = os.getcwd()
print("Current directory:", current_dir)
from num2words import num2words

from sentence_transformers import SentenceTransformer, util
from distillation.pixtral.PixtralModule import PixtralModule
from dataset.datamodule.pixtral.CustomSUNRGBDPixtralDataModule import CustomSUNRGBDPixtralDataModule
from dotenv import load_dotenv
import argparse
from transformers import LogitsProcessorList, LogitsProcessor
load_dotenv()
root_data_dir = os.getenv("root_data_dir2")
root_data_dir = os.path.abspath(root_data_dir + "data/" )
ROOT_DATA_DIR = os.path.abspath(root_data_dir)
# ROOT_DATA_DIR = os.path.abspath(root_data_dir)
current_dir = os.getcwd()
print("Current directory:", current_dir)
from collections import Counter

def remove_substring_from_path(path, substring="SUNRGBD"):
    index = path.find(substring)
    if index != -1:
        path = path[:index] + path[index + len(substring):]
    return path

def get_image_paths(img_name_rgb, img_name_depth):
    """
    Helper function to build and clean the paths for RGB and depth images.
    """
    directory = os.path.join(root_data_dir, "SUNRGBD")
    
    # Build full paths
    rgb_image_path = os.path.join(directory, img_name_rgb)
    depth_image_path = os.path.join(directory, img_name_depth)

    # Clean up the paths (e.g., remove substrings and fix slashes)
    rgb_image_path = remove_substring_from_path(rgb_image_path).replace("\\", "/")
    depth_image_path = remove_substring_from_path(depth_image_path).replace("\\", "/")

    return rgb_image_path, depth_image_path

def convert_numbers_to_words(text):
    # Check if the entire answer is a number (e.g., "1" or "42")
    if text.strip().isdigit():
        return num2words(int(text.strip()))

    # Dictionary for common single-digit mappings, for faster replacements
    number_to_word = {
        "0": "zero",
        "1": "one",
        "2": "two",
        "3": "three",
        "4": "four",
        "5": "five",
        "6": "six",
        "7": "seven",
        "8": "eight",
        "9": "nine",
        "10":"ten"
    }
    
    # # Replace single-digit numbers if found individually within text
    # for num, word in number_to_word.items():
    #     text = text.replace(f" {num} ", f" {word} ")
    #     text = text.replace(f" {num}.", f" {word}.")
    #     text = text.replace(f"{num} ", f"{word} ")
    #     text = text.replace(f" {num}", f" {word}")

    # Additional step: Convert any standalone numbers in the text to words
    words = text.split()
    for i, word in enumerate(words):
        if word.isdigit():
            words[i] = num2words(int(word))  # Convert only if the word is a digit

    return " ".join(words)

class RestrictedLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = set(allowed_token_ids)

    def __call__(self, input_ids, scores):
        mask = torch.full(scores.shape, float('-inf')).to(scores.device)
        mask[:, list(self.allowed_token_ids)] = 0  # Set allowed token positions to 0 (unmasked)
        return scores + mask
    
def get_logit_processor(answers_df ,processor ):
    
    answer_tokens = set(answers_df['Unique Answers'].str.split().sum())  # Collect unique tokens in "Answers" column

    tokenizer = processor.tokenizer
    allowed_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in answer_tokens if token in tokenizer.get_vocab()]
    logits_processor = LogitsProcessorList([RestrictedLogitsProcessor(allowed_token_ids)])

    return logits_processor

def get_model_answer(question, image, model, processor, logits_processor):
    text_prompt = question 

    # conversation = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "image"},
    #             {"type": "text", "text": text_prompt},
    #         ],
    #     },
    # ]
    
    
    # prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    additional_instructions = ".Answer in one word.Don't use the word 'based'."
    PROMPT = "<s>[INST]"+text_prompt+additional_instructions+"\n[IMG][/INST]"
    inputs = processor(images=image, text=PROMPT, return_tensors="pt").to("cuda:0", torch.float16)
    # inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda:0", torch.float32)


    pad_token_id = processor.tokenizer.eos_token_id if processor.tokenizer.pad_token_id is None else processor.tokenizer.pad_token_id  


    # autoregressively complete prompt
    with torch.no_grad():
        # output = model.generate(**inputs, max_new_tokens=100)
        # output = model.model.generate(**inputs, max_new_tokens=1, logits_processor=logits_processor, pad_token_id=pad_token_id)
        output = model.generate(**inputs, max_new_tokens=32, pad_token_id=pad_token_id)


    model_answer = processor.decode(output[0], skip_special_tokens=True)
    # print("Model Answer:", model_answer)
    
    # Isolate the answer part
    split_text = model_answer.split(".")  # Split by period to separate sections
    final_answer = split_text[-1].strip()  # Take the last part which should be the answer

    # Convert to lowercase and clean it
    extracted_answer = final_answer.lower()
    # print("Extracted Answer:", extracted_answer)

    # Convert numbers to words if needed
    final_answer = convert_numbers_to_words(extracted_answer)
    

    return final_answer

def are_colors_similar(model_answer, correct_answer, threshold=0.8):
    # Encode both answers
    model_embedding = condition_model.encode(model_answer, convert_to_tensor=True)
    correct_embedding = condition_model.encode(correct_answer, convert_to_tensor=True)

    # Compute similarity
    similarity = util.pytorch_cos_sim(model_embedding, correct_embedding).item()
    return similarity >= threshold

def convert_depth_image(depth_image_path):
    # Load depth image
    depth_image = Image.open(depth_image_path)
    depth_image_array = np.array(depth_image)

    depth_min = depth_image_array.min()
    depth_max = depth_image_array.max()

    depth_image_normalized = (255 * (depth_image_array - depth_min) / (depth_max - depth_min)).astype(np.uint8)

    depth_image_3channel_array = np.stack([depth_image_normalized] * 3, axis=-1)

    depth_image_pil = Image.fromarray(depth_image_3channel_array)

    return depth_image_pil

    
def main():
    parser = argparse.ArgumentParser(description="Pixtral OneVision Inference Script")

    # Add arguments for batch_size and checkpoint path
    parser.add_argument('--batch_size', type=int, default=1, help="Input batch size for inference (default: 1)")
    # parser.add_argument('--checkpoint_path', type=str, required=True, help="Path to the pre-trained model checkpoint")
    parser.add_argument('--subset_percentage', type=float, default=1, help="Percentage of the dataset to use (default: 1)")

    args = parser.parse_args()

    # Load the processor
    model_name = "mistral-community/pixtral-12b"
    # model_name = "mistralai/Pixtral-12B-2409"
    processor = AutoProcessor.from_pretrained(model_name)

    # Initialize the model
    model = PixtralModule(model_name, processor)
    # model.load_state_dict(torch.load(args.checkpoint_path, map_location=torch.device('cpu')))
    model.model.eval()  # Set the model to evaluation mode
    model.model.to("cuda:0")

    
    print("Pixtral model loaded successfully")

    # Initialize the data module
    # data_module = CustomSUNRGBDPixtralDataModule(
    #     root_data_dir=ROOT_DATA_DIR,
    #     processor=processor,
    #     batch_size=args.batch_size,
    #     num_workers=4,
    #     subset_percentage=args.subset_percentage,
    # )


    # CSV_FILE_DIR = "dataset/SUNRGBD_Dataset"
    csv_file_path_val = os.path.join(ROOT_DATA_DIR, "SUNRGBD/csv_data", "val_dataset.csv")
    csv_file_path_test = os.path.join(ROOT_DATA_DIR, "SUNRGBD/csv_data", "test_dataset.csv")

    df_val = pd.read_csv(csv_file_path_val)
    df_test = pd.read_csv(csv_file_path_test)



    results = []



    gts_type = "val"
    if gts_type== "val":
        df_val = pd.read_csv(csv_file_path_val)
        df = df_val
        ds_len = len(df_val)

    else:
        df_test = pd.read_csv(csv_file_path_test)
        df = df_test
        ds_len = len(df_test)


    # new code
    # ds_len = 1000
    incorrect = 0
    correct = 0
    # Specify the directory where images will be saved
    output_dir = "dataset/SUNRGBD_Dataset/sample_check/incorrect_predictions/"
    os.makedirs(output_dir, exist_ok=True)
    condition_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    unique_tokens_path = os.path.join(ROOT_DATA_DIR, "SUNRGBD/", "unique_tokens_new4.csv")
    answers_df = pd.read_csv(unique_tokens_path)

    logits_processor = get_logit_processor(answers_df, processor)

    incorrect_categories = []
    incorrect_df = pd.DataFrame(columns=['Question_Id', 'Questions', 'Question_Type', 'Answers',"Model_Answer"])
    data_list = []
    predictions = []
    references = []

    for i in tqdm(range(ds_len), desc="Processing Results"):
        try:
            question_id = int(df.iloc[i]['Question_Id'])
            question = df.iloc[i]['Questions']
            question_type = df.iloc[i]['Question_Type']
            answer_type = df.iloc[i]['Question_Type']
            answers = df.iloc[i]['Answers']

            rgb_image_path = df.iloc[i]['Image_Path']
            depth_image_path = df.iloc[i]['Depth_Path']

            rgb_image_path, depth_image_path = get_image_paths(rgb_image_path, depth_image_path)

            rgb_image = Image.open(rgb_image_path)
            # rgb_image_np = np.array(rgb_image)



            # depth_image = convert_depth_image(depth_image_path)
            # depth_image_array = np.array(depth_image)

            model_final_answer = get_model_answer(question, rgb_image , model.model, processor, logits_processor)
            # print( "Question:", question   ,"Final Answer:", model_final_answer , ";Ground Truth:", answers)



            # row_data = {
            #     'Question_Id': question_id,
            #     'Questions': question,
            #     'Question_Type': question_type,
            #     'Answers': answers,
            #     'Model_Answer':model_final_answer
            # }

            data_list.append({
                'Question_Id': question_id,
                'Questions': question,
                'Question_Type': question_type,
                'Answers': answers,
                'Model_Answer': model_final_answer
            })

            predictions.append(model_final_answer)
            references.append(answers)



            # #For Calculating Accuracy
            # if (model_final_answer !="no") and (len(model_final_answer) == 1 or len(model_final_answer) == 2):

            #     incorrect_df.loc[len(incorrect_df)] = row_data


            #     incorrect +=1


            # elif model_final_answer.strip() not in answers.split():
            #     # print("final_answer:", model_final_answer)
            #     # print("answers:", answers)
            #     incorrect_df.loc[len(incorrect_df)] = row_data

            #     incorrect +=1
            #     # print("Incorrect")

            # else:
            #     correct +=1
            #     # print("Correct")

        except Exception as e:
            print("Error:", e)
            continue

    
    #Write the results to a CSV file
    results_df = pd.DataFrame(data_list)
    
    path_to_save = os.path.join( "dataset/predictions/", "results_"+gts_type+"_"+"pixtral"+".csv")
    
    results_df.to_csv(path_to_save, index=False)  
    print("Results saved to:", path_to_save)


    #     results.append({
    #         "question_id": question_id,
    #         "answer": model_final_answer
    #     })
        
        






    # incorrect_df.to_csv("dataset/SUNRGBD_Dataset/sample_check/incorrect_predictions/"+'incorrect_predictions_pixtral12b.csv', index=False)
    # print("Incorrect predictions saved")
    # category_counts = Counter(incorrect_categories)
    # sorted_categories = category_counts.most_common()
    # print("Incorrectly classified question types from highest to lowest frequency:")
    # for question_type, count in sorted_categories:
    #     print(f"{question_type}: {count}")

    # print(f"Accuracy: {correct/ds_len}")




    # data_module.setup(stage="test")


    # # Get the test dataloader
    # test_dataloader = data_module.test_dataloader()

    # # Perform inference
    # results = []
    # with torch.no_grad():  # Disable gradient computation for inference
    #     for batch in test_dataloader:
    #         input_ids = batch["depth_input_ids"]
    #         rgb_pixel_values = batch["rgb_pixel_values"]
    #         depth_pixel_values = batch["depth_pixel_values"]
    #         labels = batch["labels"]
    #         question_id = batch["question_id"]
            
    #         # Forward pass
    #         outputs = model(
    #             input_ids=input_ids,
    #             rgb_pixel_values=rgb_pixel_values,
    #             depth_pixel_values=depth_pixel_values,
    #             labels=None  # No labels needed during inference
    #         )
    #         # print(outputs)
    #         # print("Worked")
            
    #         # Decode predictions
    #         predicted_ids = outputs.logits.argmax(dim=-1)
    #         decoded_output = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    #         print(decoded_output)
            
    #         # Collect results
    #         results.extend(decoded_output)

    # # Save results to a file
    # output_file = os.path.join(current_dir, "inference_results.txt")
    # with open(output_file, "w") as f:
    #     for line in results:
    #         f.write(f"{line}\n")

    # print(f"Inference completed. Results saved to {output_file}")


if __name__ == "__main__":
    main()
