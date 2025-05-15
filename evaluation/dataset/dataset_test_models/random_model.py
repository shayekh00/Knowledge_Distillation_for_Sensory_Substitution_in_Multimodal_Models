import pandas as pd
import random
import os
pd.set_option('display.max_colwidth', None)  # None removes the limit, or you can set a specific limit\\
import matplotlib.pyplot as plt
import numpy as np

def plot_accuracy_vs_expected(question_types, accuracy, expected_accuracy, save_path='accuracy_vs_expected_accuracy.png'):
    """
    Creates a bar plot comparing accuracy vs expected accuracy by question type and saves it to the specified path.
    
    Args:
    question_types (list): List of question types.
    accuracy (list): List of accuracy percentages.
    expected_accuracy (list): List of expected accuracy percentages.
    save_path (str): File path where the plot will be saved.
    """
    # Bar width
    bar_width = 0.35

    # Set the positions of the bars on the x-axis
    index = np.arange(len(question_types))

    # Create a bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(index, accuracy, bar_width, label='Accuracy')
    plt.bar(index + bar_width, expected_accuracy, bar_width, label='Expected Accuracy')

    # Add labels and title
    plt.xlabel('Question Type')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs Expected Accuracy by Question Type')

    # Add tick labels for x-axis
    plt.xticks(index + bar_width / 2, question_types, rotation=45, ha='right')

    # Add a legend
    plt.legend()

    # Adjust layout for better spacing
    plt.tight_layout()

    # Save the plot to the specified file path
    plt.savefig(save_path)

    # Display the plot
    plt.show()


def random_model(df):
    """
    Randomly selects an answer for each question based on the Question_Type and computes accuracy for each type.

    Args:
        df (pd.DataFrame): The dataset with questions, answers, and question types.

    Returns:
        dict: A dictionary containing accuracy and expected accuracy for each Question_Type.
    """
    # Step 1: Build a dictionary of possible unique answers for each Question_Type
    unique_answers_by_type = df.groupby('Question_Type')['Answers'].unique().to_dict()

    # Create a column to store predicted answers
    df['Predicted_Answer'] = None

    # Step 2: Iterate over each row and randomly select an answer for each question
    for index, row in df.iterrows():
        question_type = row['Question_Type']
        possible_answers = unique_answers_by_type[question_type]
        
        # Select a random answer from the possible answers for this Question_Type
        predicted_answer = random.choice(possible_answers)
        
        # Assign the predicted answer to the DataFrame
        df.at[index, 'Predicted_Answer'] = predicted_answer

    # Step 3: Compute accuracy and expected accuracy for each Question_Type
    accuracy_by_type = {}
    
    for question_type, group in df.groupby('Question_Type'):
        total_questions = len(group)
        correct_predictions = (group['Answers'] == group['Predicted_Answer']).sum()
        accuracy = correct_predictions / total_questions
        
        # Calculate expected accuracy
        num_unique_answers = len(unique_answers_by_type[question_type])
        expected_accuracy = (1 / num_unique_answers) * 100 if num_unique_answers > 0 else 0
        
        accuracy_by_type[question_type] = {
            'Accuracy': accuracy,
            'Expected Accuracy': expected_accuracy
        }

    return accuracy_by_type



# Directory path where your CSV file is located
directory_path = 'dataset/SUNRGBD_Dataset/SUNRGBD/csv_data/'
# CSV file name
csv_file_name = 'train_dataset.csv'
# Full file path
file_path = os.path.join(directory_path, csv_file_name)

# Read CSV file using Pandas
df = pd.read_csv(file_path)

accuracy_by_question_type = random_model(df)
for question_type, metrics in accuracy_by_question_type.items():
        # print(f"Question Type: {question_type}, Accuracy: {metrics}")
        accuracy = metrics['Accuracy']* 100
        expected_accuracy = metrics['Expected Accuracy']
        print(f"Question Type: {question_type}, Accuracy: {accuracy:.2f}%, Expected Accuracy: {expected_accuracy:.2f}%")



question_types = ["Color Identification", "Count", "Direction", "Object Identification", "Proximity", "Yes/No"]
plot_accuracy_vs_expected(question_types, accuracy, expected_accuracy, 'results/accuracy_vs_expected_accuracy2.png')