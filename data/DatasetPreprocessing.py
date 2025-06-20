# Importing Libraries
from datasets import load_dataset
from dotenv import load_dotenv
import kagglehub
import os
import random

load_dotenv()

# Loading HF_TOKEN from .env file
token = os.getenv("HF_TOKEN")

# Loading the Datasets for different tasks
LMData = kagglehub.dataset_download("rtatman/state-of-the-union-corpus-1989-2017") # Used for Language Modelling the Decoder-Only Model
ChatData = load_dataset("li2017dailydialog/daily_dialog", trust_remote_code=True, token=token) # Used for Fine-Tuning the Decoder-Only Model

# Combining Language Modelling Data
data = []
for filename in os.listdir(LMData):
    if filename.endswith('.txt'):
        with open(os.path.join(LMData, filename), "r", encoding="utf-8") as file:
            text = file.read().strip()
            if text:
                data.append(text)
                
# Shuffling Data for Generalization
random.shuffle(data)

split_idx = int(0.9 * len(data))
train_data = data[:split_idx]
valid_data = data[split_idx:]

# Combine into text
LMDataTrain = "\n\n".join(train_data) # 90% Train Data
LMDataValid = "\n\n".join(valid_data) # 10% Valid Data

# Saving Language Modelling Data
def format_lm_data(type, value):
    with open(f'NeuroFormer/data/raw/Language Modelling/{type}.txt', 'w', encoding='utf-8') as f:
        f.write(value)

format_lm_data('train', LMDataTrain)
format_lm_data('valid', LMDataValid)

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