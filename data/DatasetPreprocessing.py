# Importing Libraries
from datasets import load_dataset
from dotenv import load_dotenv
import os

load_dotenv()

# Loading HF_TOKEN from .env file
token = os.getenv("HF_TOKEN")

# Loading the Datasets for different tasks
ChatData = load_dataset("li2017dailydialog/daily_dialog", trust_remote_code=True, token=token) # Used for Fine-Tuning the Decoder-Only Model

# Saving Chatbot Data in txt format 
def format_chat_data(type, value):
    with open(f"NeuroFormer/data/raw/{type}.txt", "w", encoding="utf-8") as f:
        for dialog in value:
            for i, utterance in enumerate(dialog):
                speaker_tag = "<user>" if i % 2 == 0 else "<assistant>"
                f.write(f"{speaker_tag} {utterance.strip()}\n")
            f.write("\n")  # Blank line to separate dialogs

# Splitting Dataset into Train, Validation, Test set
ChatDataTrain = ChatData["train"]["dialog"] + ChatData["test"]["dialog"] # 90% Train Data
ChatDataVal = ChatData["validation"]["dialog"] # 10% Validation Data

format_chat_data('train', ChatDataTrain)
format_chat_data('valid', ChatDataVal)