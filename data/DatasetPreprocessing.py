# Importing Libraries
from datasets import load_dataset
from dotenv import load_dotenv
from itertools import islice
import os
import re

load_dotenv()

# Loading HF_TOKEN from .env file
token = os.getenv("HF_TOKEN")

# Loading the Datasets for different tasks
LMData = load_dataset("Skylion007/openwebtext", trust_remote_code=True, split='train', streaming=True) # Used for Pretraining the Decocder-Only Model
LMData = list(islice(LMData, 100000))  
ChatData = load_dataset("li2017dailydialog/daily_dialog", trust_remote_code=True, token=token) # Used for Training the Decoder-Only Model

def clean_text(text):
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"\s+", " ", text).strip()
    text = text.encode("ascii", "ignore").decode()
    return re.sub(r"[^a-zA-Z0-9\s.,!?;:'\"()\-\n]", "", text)

def format_lm_data(type, value):
    with open(f"NeuroFormer/data/pretrain/{type}.txt", "w", encoding="utf-8") as f:
        for item in value:
            text = item['text'].strip().replace("\n", " ")
            text = clean_text(text)
            if len(text) > 50:
                f.write(text + "\n")

split = int(0.9 * len(LMData))
LMTrain = LMData[ : split]
LMVal = LMData[split : ]

format_lm_data('train', LMTrain)
format_lm_data('valid', LMVal)

# Saving Chatbot Data in txt format 
def format_chat_data(type, value):
    with open(f"NeuroFormer/data/finetune/{type}.txt", "w", encoding="utf-8") as f:
        for dialog in value:
            for i, utterance in enumerate(dialog):
                cleaned = clean_text(utterance.strip())
                if cleaned:
                    tag = "<user>" if i % 2 == 0 else "<assistant>"
                    f.write(f"{tag} {cleaned}\n")
            f.write("\n")  # Blank line to separate dialogs

# Splitting Dataset into Train, Validation Set
ChatDataTrain = ChatData["train"]["dialog"] + ChatData["test"]["dialog"] # 90% Train Data
ChatDataVal = ChatData["validation"]["dialog"] # 10% Validation Data

format_chat_data('train', ChatDataTrain)
format_chat_data('valid', ChatDataVal)