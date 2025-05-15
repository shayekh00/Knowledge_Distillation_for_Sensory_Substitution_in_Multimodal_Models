import os
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class VQADataset(Dataset):
    def __init__(self, annotation_file, question_file, img_dir):
        """
        VQADataset initializes the dataset with annotations, questions, and images.
        
        :param annotation_file: Path to the VQA annotations JSON file.
        :param question_file: Path to the VQA questions JSON file.
        :param img_dir: Directory containing the images.
        """
        # Load annotation and question JSON files
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)["annotations"]
        
        with open(question_file, 'r') as f:
            self.questions = json.load(f)["questions"]
        
        self.img_dir = img_dir

        # Mapping questions by their question_id for faster lookup
        self.question_map = {item["question_id"]: item["question"] for item in self.questions}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        Return a single data point (image, question, answers).
        
        :param idx: Index of the item in the dataset.
        :return: A dictionary with image, question, and answers.
        """
        # Load the annotation entry
        annotation = self.annotations[idx]
        image_id = annotation['image_id']
        question_id = annotation['question_id']
        question_type = annotation['question_type']
        answer_type = annotation['answer_type']
        # answers = [ans['answer'] for ans in annotation['answers']]
        answers = annotation['answers']
        multiple_choice_answer = annotation['multiple_choice_answer']

        # Load the image
        image_path = os.path.join(self.img_dir, f'COCO_val2014_{image_id:012d}.jpg')
        print(image_path)
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)

        # Get the corresponding question
        question = self.question_map[question_id]

        # Return data point
        return {
            'image': image,
            'question': question,
            'answers': answers,
            'multiple_choice_answer': multiple_choice_answer,
            'question_id': question_id,
            'image_id': image_id,
            'question_type': question_type,
            'answer_type': answer_type
        }