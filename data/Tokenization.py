# Importing Libraries
import torch
import json
from transformers import GPT2Tokenizer, BertTokenizer, T5Tokenizer

# A Simple Tokenizer Class that handles Decoder-Only, Encoder-Only, Encoder-Decoder tasks
class Tokenizer:
    
    def __init__(self):
        
        # Initializing all Tokenizers
        self.gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
        
        # Adding Special Tokens to Chatbot Tokenizer
        self.gpt_tokenizer.add_special_tokens({
            'pad_token' : '<pad>',
            'bos_token' : '<bos>',
            'eos_token' : '<eos>',
            'additional_special_tokens' : ['<user>', '<assistant>']
        })
        
        # Adding Special Tokens to Ques Ans Tokenizer
        self.bert_tokenizer.add_special_tokens({
            'additional_special_tokens' : ['<context>', '<question>', '<answer>']
        })
        
        # Adding Special Tokens to Translation Tokenizer
        self.t5_tokenizer.add_special_tokens({
            'additional_special_tokens' : ['<source>', '<target>']            
        })
        
        # Saving Modified Tokenizers
        self.gpt_tokenizer.save_pretrained('NeuroFormer/tokenizers/gpt2')
        self.bert_tokenizer.save_pretrained('NeuroFormer/tokenizers/bert')
        self.t5_tokenizer.save_pretrained('NeuroFormer/tokenizers/t5')
        
    # Tokenizing Language Modelling Data
    def tokenize_lm(self):
        
        input_path = "NeuroFormer/data/raw/Language Modelling/train.txt"
        output_path = "NeuroFormer/data/tokenized/Language Modelling/train.pt"
        tokenized_data = []
        
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()
            
        for line in lines:
            line = line.strip()
            if len(line) > 10:
                line = f"<bos> {line} <eos>"
                tokens = self.gpt_tokenizer(
                    line,
                    max_length = 512,
                    padding = 'max_length',
                    truncation = True,
                    return_tensors = 'pt' 
                )
                tokenized_data.append({
                    'input_ids' : tokens['input_ids'].squeeze(0),
                    'attention_mask' : tokens['attention_mask'].squeeze(0)
                })
            
        torch.save(tokenized_data, output_path)
        print(f"saved {len(tokenized_data)} Language Modelling Samples")
        
    # Tokenizing Chatbot Data
    def tokenize_chatbot_data(self):
        
        for split in ['train', 'valid', 'test']:
            
            input_path = f"NeuroFormer/data/raw/Chatbot Data/{split}.txt"
            output_path = f"NeuroFormer/data/tokenized/Chatbot Data/{split}.pt"
            tokenized_data = []
            
            with open(input_path, 'r', encoding='utf-8') as f:
                conversations = f.read().strip().split('\n\n')
            
            for conv in conversations:
                if len(conv) > 20:
                    conv = f"<bos> {conv.strip()} <eos>"
                    tokens = self.gpt_tokenizer(
                        conv,
                        max_length = 512,
                        padding = 'max_length',
                        truncation = True,
                        return_tensors = 'pt'
                    )
                    tokenized_data.append({
                        'input_ids' : tokens['input_ids'].squeeze(0),
                        'attention_mask' : tokens['attention_mask'].squeeze(0)
                    })
            
            torch.save(tokenized_data, output_path)
            print(f"saved {len(tokenized_data)} Chatbot {split} Samples")
    
    def encode_chatbot_input(self, text: str) -> torch.Tensor:
        return self.gpt_tokenizer.encode(text, return_tensors = 'pt')
    
    def decode_chatbot_output(self, tokens: torch.Tensor) -> str:
        return self.gpt_tokenizer.decode(tokens, skip_special_tokens=True)
    
    # Tokenizing Ques Ans Data
    def tokenize_qa_data(self):
        
        for split in ['train', 'valid', 'test']:
            
            input_path = f"NeuroFormer/data/raw/QuesAns Data/{split}.json"
            output_path = f"NeuroFormer/data/tokenized/QuesAns Data/{split}.pt"
            tokenized_data = []
            
            with open(input_path, 'r', encoding='utf-8') as f:
                items = json.load(f)
                
            for item in items:
                input_text = f"<context> {item['context']} <question> {item['question']}"
                answer_text = f"<answer> {item['answer']}"
                
                input_tokens = self.bert_tokenizer(
                    input_text,
                    max_length = 512,
                    padding = 'max_length',
                    truncation = True,
                    return_tensors = 'pt'
                )
                answer_tokens = self.bert_tokenizer(
                    answer_text,
                    max_length = 64,
                    padding = 'max_length',
                    truncation = True,
                    return_tensors = 'pt'
                )
                
                tokenized_data.append({
                'input_ids': input_tokens['input_ids'].squeeze(0),
                'attention_mask': input_tokens['attention_mask'].squeeze(0),
                'answer_ids': answer_tokens['input_ids'].squeeze(0),
                'answer_mask': answer_tokens['attention_mask'].squeeze(0)
                })
                
            torch.save(tokenized_data, output_path)
            print(f"saved {len(tokenized_data)} QuesAns {split} Samples")
    
    def encode_qa_input(self, context: str, question: str) -> torch.Tensor:
        input_text = f"<context> {context} <question> {question}"
        return self.bert_tokenizer.encode(input_text, return_tensors = 'pt')
    
    def decode_qa_output(self, tokens: torch.Tensor) -> str:
        return self.bert_tokenizer.decode(tokens, skip_special_tokens=True)
            
    # Tokenizing Translation Data
    def tokenize_translation_data(self):
        
        for split in ['train', 'valid', 'test']:
            
            input_path = f"NeuroFormer/data/raw/Translation Data/{split}.json"
            output_path = f"NeuroFormer/data/tokenized/Translation Data/{split}.pt"
            tokenized_data = []
            
            with open(input_path, 'r', encoding='utf-8') as f:
                items = json.load(f)
            
            for item in items:
                source_text = f"<source> {item['source']}"
                target_text = f"<target> {item['target']}"

                source_tokens = self.t5_tokenizer(
                    source_text,
                    max_length=256,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                target_tokens = self.t5_tokenizer(
                    target_text,
                    max_length=256,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                
                tokenized_data.append({
                'input_ids': source_tokens['input_ids'].squeeze(0),
                'attention_mask': source_tokens['attention_mask'].squeeze(0),
                'labels': target_tokens['input_ids'].squeeze(0),
                'decoder_attention_mask': target_tokens['attention_mask'].squeeze(0)
                })
            
            torch.save(tokenized_data, output_path)
            print(f"saved {len(tokenized_data)} Translation {split} Samples")
    
    def encode_translation_input(self, source: str) -> torch.Tensor:
        return self.t5_tokenizer.encode(f"<source> {source}", return_tensors="pt")

    def decode_translation_output(self, tokens: torch.Tensor) -> str:
        return self.t5_tokenizer.decode(tokens, skip_special_tokens=True)
            
    def tokenize_all(self):
        print('Tokenization Started')
        self.tokenize_lm()
        self.tokenize_chatbot_data()
        self.tokenize_qa_data()
        self.tokenize_translation_data()
        print('Tokenization Done')
        
tokenizer = Tokenizer()
tokenizer.tokenize_all()

encoded_chat = tokenizer.encode_chatbot_input("<user> Hello! <assistant> Hi there!")
print(f"Encoded Chatbot Input: {encoded_chat}")
decoded_chat = tokenizer.decode_chatbot_output(encoded_chat[0])
print(f"Decoded Chatbot Output: {decoded_chat}")  

encoded_qa = tokenizer.encode_qa_input("<context> AI is artificial intelligence.", "<question> What is AI?")
print(f"Encoded QuesAns Input: {encoded_qa}")
decoded_qa = tokenizer.decode_qa_output(encoded_qa[0])
print(f"Decoded Chatbot Output: {decoded_qa}")

encoded_trans = tokenizer.encode_translation_input("<source> How are you?")
print(f"Encoded Translation Input: {encoded_trans}")
decoded_trans = tokenizer.decode_translation_output(encoded_trans[0])
print(f"Decoded Translation Output: {decoded_trans}")