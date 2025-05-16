import pandas as pd
import re
from num2words import num2words
from textblob import Word
from transformers import pipeline
import torch
fix_spelling = pipeline("text2text-generation",model="oliverguhr/spelling-correction-english-base",device=0 if torch.cuda.is_available() else -1)


combined_words_df_path = "combined_words_mapping.csv"
combined_words = pd.read_csv(combined_words_df_path).set_index('Combined Word').to_dict()['Separated Word']
print("Combined Words Loaded")

# Function to fix combined words, convert numbers, remove trailing numbers, and lowercase all words
def process_answers_column(df):
    def process_text(text):

        # Ensure text is a string
        text = str(text) if text is not None else ""


        for combined, fixed in combined_words.items():
            text = re.sub(rf"\b{combined}\b", fixed, text, flags=re.IGNORECASE)
        
        # Step 2: Convert any numbers to words
        text = re.sub(r"\b\d+\b", lambda match: num2words(int(match.group())), text)

        # Step 3: Remove trailing numbers from words
        text = re.sub(r"\b(\w+?)(\d+)\b", r"\1", text)

        # Step 4: Convert to lowercase
        text = text.lower()

        return text

    # Apply the process_text function to each row in the 'Answers' column
    df['Answers'] = df['Answers'].apply(process_text)
    return df




def process_text_only(text):
    # Ensure text is a string
    text = str(text) if text is not None else ""
    
    # Step 1: Replace combined words with separated words
    for combined, fixed in combined_words.items():
        text = re.sub(rf"\b{combined}\b", fixed, text, flags=re.IGNORECASE)
    
    # Step 2: Convert any numbers to words
    text = re.sub(r"\b\d+\b", lambda match: num2words(int(match.group())), text)
    
    # Step 3: Remove trailing numbers from words
    text = re.sub(r"\b(\w+?)(\d+)\b", r"\1", text)
    
    # Step 4: Correct spelling errors
    text = " ".join([Word(word).correct() for word in text.split()])

    # Step 4: Correct spelling errors using llm
    text = fix_spelling( text ,max_length=20)[0]["generated_text"]
    text = text.rstrip(".")
    
    # Step 5: Convert to lowercase
    text = text.lower()
    
    return text
