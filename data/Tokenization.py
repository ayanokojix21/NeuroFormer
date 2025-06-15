# Importing Libraries
import torch
import json
from transformers import GPT2Tokenizer, BertTokenizerFast, T5Tokenizer

# A Simple Tokenizer Class that handles Decoder-Only, Encoder-Only, Encoder-Decoder tasks
class Tokenizer:
    
    def __init__(self):
        
        # Initializing all Tokenizers
        self.gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.bert_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
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
        if len(tokens.shape) > 1:
            tokens = tokens[0]
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
                
                context = item['context']
                question = item['question']
                answer_text = item['answer']

                # Find character start and end index of answer in context
                start_char = context.find(answer_text)
                if start_char == -1:
                    continue  # answer not found in context, skip

                end_char = start_char + len(answer_text)

                # Tokenize context and question using the pair encoding method
                encoding = self.bert_tokenizer(
                    context,
                    question,
                    max_length=512,
                    padding='max_length',
                    truncation='only_first',  # truncate context only
                    return_offsets_mapping=True,
                    return_tensors='pt'
                )

                input_ids = encoding['input_ids'].squeeze(0)
                attention_mask = encoding['attention_mask'].squeeze(0)
                offsets = encoding['offset_mapping'].squeeze(0)

                # Find start and end token positions
                start_idx, end_idx = -1, -1
                for idx, (start, end) in enumerate(offsets):
                    if start <= start_char < end:
                        start_idx = idx
                    if start < end_char <= end:
                        end_idx = idx

                # If couldn't find proper span, skip the example
                if start_idx == -1 or end_idx == -1:
                    continue

                tokenized_data.append({
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'start_idx': torch.tensor(start_idx),
                    'end_idx': torch.tensor(end_idx)
                })                              
                
            torch.save(tokenized_data, output_path)
            print(f"saved {len(tokenized_data)} QuesAns {split} Samples")
    
    def encode_qa_input(self, context: str, question: str) -> dict:
        return self.bert_tokenizer(
            context,
            question,
            max_length=512,
            padding='max_length',
            truncation='only_first',
            return_tensors='pt'
        )

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
                    max_length=512,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                target_tokens = self.t5_tokenizer(
                    target_text,
                    max_length=512,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                
                labels = target_tokens['input_ids'].squeeze(0)
                labels[labels == self.t5_tokenizer.pad_token_id] = -100

                tokenized_data.append({
                'input_ids': source_tokens['input_ids'].squeeze(0),
                'attention_mask': source_tokens['attention_mask'].squeeze(0),
                'labels': labels,
                'decoder_attention_mask': target_tokens['attention_mask'].squeeze(0)
                })
            
            torch.save(tokenized_data, output_path)
            print(f"saved {len(tokenized_data)} Translation {split} Samples")
    
    def encode_translation_input(self, source: str) -> dict:
        return self.t5_tokenizer(
            f"<source> {source}",
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        )

    def decode_translation_output(self, tokens: torch.Tensor) -> str:
        output = self.t5_tokenizer.decode(tokens, skip_special_tokens=True)
        output = output.replace("<source>", "").replace("<target>", "").strip()
        return output
            
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

encoded_qa = tokenizer.encode_qa_input("AI is artificial intelligence.", "What is AI?")
print(f"Encoded QuesAns Input: {encoded_qa}")
decoded_qa = tokenizer.decode_qa_output(encoded_qa['input_ids'][0])
print(f"Decoded QuesAns Output: {decoded_qa}")

encoded_trans = tokenizer.encode_translation_input("<source> How are you?")
print(f"Encoded Translation Input: {encoded_trans}")
decoded_trans = tokenizer.decode_translation_output(encoded_trans['input_ids'][0])
print(f"Decoded Translation Output: {decoded_trans}")