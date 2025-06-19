# Importing Libraries
from datasets import load_dataset
from dotenv import load_dotenv
import os
import re 

load_dotenv()

# Loading HF_TOKEN from .env file
token = os.getenv("HF_TOKEN")

# Loading the Datasets for different tasks
LMData = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", token=token) # Used for Language Modelling the Decoder-Only Model
ChatData = load_dataset("li2017dailydialog/daily_dialog", trust_remote_code=True, token=token) # Used for Fine-Tuning the Decoder-Only Model

# Cleaning Raw Data for Language Modelling
def clean_line(line):
    line = line.strip()

    if not line:
        return ""
    
    # Normalizing to lowercase for metadata filtering
    lower_line = line.lower()

    if re.match(r"^=+\s?.*?\s?=+$", line):  # Removing Wikipedia Headers
        return ""
 
    if line.startswith(("[[", "{{", "'''", "```", "File:", "Image:")): 
        return ""

    if re.match(r"^\[https?://", line) or "http" in line: # Removing Website Headers
        return ""

    if lower_line.startswith(("category:", "thumb", "special:", "help:", "user:")):
        return ""

    if len(line.split()) < 50: # Removing Short Lines
        return ""
    
    # Removing MetaData from Lines
    line = re.sub(r"[\*\[\]{}<>\\|@#]", "", line)
    line = re.sub(r"\s+", " ", line)

    return line

# Putting Cleaned Lines in a list to store them for tokenization training
cleaned_lines = []
for split in ['train', 'validation', 'test']:
    for x in LMData[split]:
        cleaned = clean_line(x['text'])
        if cleaned:
            cleaned_lines.append(cleaned)

# Splitting the lines in Train And Validation set
LMData = cleaned_lines
split = int(len(cleaned_lines) * 0.9)
LMDataTrain = LMData[ : split]
LMDataVal = LMData[split : ]

# Saving Cleaned Lines for Language Modelling
def format_lm_data(type, value):
    with open(f"NeuroFormer/data/raw/Language Modelling/{type}.txt", "w", encoding="utf-8") as f:
        for line in value:
            f.write(line + "\n")
            
format_lm_data('train', LMDataTrain)
format_lm_data('valid', LMDataVal)

# Saving Chatbot Data in txt format 
def format_chat_data(type, value):
    with open(f"NeuroFormer/data/raw/Chatbot Data/{type}.txt", "w", encoding="utf-8") as f:
        for dialog in value:
            for i, utterance in enumerate(dialog):
                speaker_tag = "<user>" if i % 2 == 0 else "<assistant>"
                f.write(f"{speaker_tag} {utterance.strip()}\n")
            f.write("\n")  # Blank line to separate dialogs

# Splitting Dataset into Train, Validation, Test set
ChatDataTrain = ChatData["train"]["dialog"] # 80% Train Data
ChatDataVal = ChatData["validation"]["dialog"] # 10% Validation Data
ChatDataTest = ChatData["test"]["dialog"] # 10% Test Data 

format_chat_data('train', ChatDataTrain)
format_chat_data('valid', ChatDataVal)
format_chat_data('test', ChatDataTest)