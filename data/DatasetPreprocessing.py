# Importing Libraries
from datasets import load_dataset
from dotenv import load_dotenv
import os
import re 
import json

load_dotenv()

# Loading HF_TOKEN from .env file
token = os.getenv("HF_TOKEN")

# Loading the Datasets for different tasks
LMData = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", token=token) # Used for Language Modelling the Decoder-Only Model
ChatData = load_dataset("li2017dailydialog/daily_dialog", trust_remote_code=True, token=token) # Used for Fine-Tuning the Decoder-Only Model
TranslationData = load_dataset("Aarif1430/english-to-hindi", token=token) # Used For Fine-Tuning Encoder-Decoder Model
QuesAnsData = load_dataset("rajpurkar/squad", token=token) # Used For Fine-Tuning Encoder-Only Model

# Cleaning Raw Data for Language Modelling
def clean_line(line: str) -> str:
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

    if len(line.split()) < 4: # Removing Short Lines
        return ""
    
    # Removing MetaData from Lines
    line = re.sub(r"[\*\[\]{}<>\\|@#]", "", line)
    line = re.sub(r"\s+", " ", line)

    return line

# Putting Cleaned Lines in a list to store them for tokenization
cleaned_lines = []
for split in ['train', 'validation', 'test']:
    for x in LMData[split]:
        cleaned = clean_line(x['text'])
        if cleaned:
            cleaned_lines.append(cleaned)

# Saving Cleaned Lines for Language Modelling
with open("NeuroFormer/data/raw/Language Modelling/train.txt", "w", encoding="utf-8") as f:
    for line in cleaned_lines:
        f.write(line + "\n")

# Converting the Dialogs in a Single Line 
def flatten_conversation(dialog, user_token="<user>", assistant_token="<assistant>"):
    result = ""
    for i, line in enumerate(dialog):
        speaker = user_token if i % 2 == 0 else assistant_token
        result += f"{speaker} {line.strip()}\n"
    return result.strip()

ChatData = ChatData["train"]["dialog"]
ChatDataTrain = ChatData[ : 3200] # 80% Train Data
ChatDataVal = ChatData[3200 : 3600] # 10% Validation Data
ChatDataTest = ChatData[3600 : 4000] # 10% Test Data 

# Saving Lines for Fine-Tuning
def format_chat_data(type, value):
    with open(f"NeuroFormer/data/raw/Chatbot Data/{type}.txt", "w", encoding="utf-8") as f:
        for dialog in value:
            flat = flatten_conversation(dialog)
            f.write(flat + "\n\n")  # blank line between dialogs

format_chat_data('train', ChatDataTrain)
format_chat_data('valid', ChatDataVal)
format_chat_data('test', ChatDataTest)

# Formatting Translation Data in Json Format
def format_translation_data():
    formatted_data = []
    for sample in TranslationData['train']:
        en = sample["english_sentence"].strip()
        hi = sample["hindi_sentence"].strip()
        if en and hi:
            formatted_data.append({
                "source": en,
                "target": hi
            })
    return formatted_data

TranslationData = format_translation_data()
TranslationDataTrain = TranslationData[ : 6400] # 80% Train Data
TranslationDataVal = TranslationData[6400 : 7200] # 10% Validation Data
TranslationDataTest = TranslationData[7200 : 8000] # 10% Test Data

# Converting into Json File
def convert_to_json_trans(type, value):
    with open(f"NeuroFormer/data/raw/Translation Data/{type}.json", "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)

convert_to_json_trans('train', TranslationDataTrain)
convert_to_json_trans('valid', TranslationDataVal)
convert_to_json_trans('test', TranslationDataTest)


# Formatting QuesAnsData into Json Format
def formatting_qa_data():
    formatted_data = []
    for sample in QuesAnsData['train']:
        context = sample["context"].strip()
        question = sample["question"].strip()
        answers = sample["answers"]["text"]
        
        # Use first answer only (for simplicity)
        if context and question and answers:
            formatted_data.append({
                "context": context,
                "question": question,
                "answer": answers[0].strip()
            })
    
    return formatted_data

QuesAnsData = formatting_qa_data()
QuesAnsDataTrain = QuesAnsData[ : 3200] # 80% Train Data 
QuesAnsDataVal = QuesAnsData[3200 : 3600] # 10% Validation Data 
QuesAnsDataTest = QuesAnsData[3600: 4000] # 10% Test Data

def convert_to_json_qa(type, value):
    with open(f"NeuroFormer/data/raw/QuesAns Data/{type}.json", "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)

convert_to_json_qa('train', QuesAnsDataTrain)
convert_to_json_qa('valid', QuesAnsDataVal)
convert_to_json_qa('test', QuesAnsDataTest)