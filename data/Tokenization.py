# Importing Libraries
import torch
from transformers import PreTrainedTokenizerFast

# Tokenizers Path
tokenizer_path = 'NeuroFormer/tokenizers/tokenizer'

# A Simple Tokenizer Class that handles Decoder-Only tasks
class Tokenizer:
    
    def __init__(self):
        
        # Initializing all Tokenizers
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        
    # Tokenizing Language Modelling Data
    def tokenize_lm(self):
        
        for split in ['train', 'valid']:
        
            input_path = f"NeuroFormer/data/raw/Language Modelling/{split}.txt"
            output_path = f"NeuroFormer/data/tokenized/Language Modelling/{split}.pt"
            input_ids = []
            attention_mask = []
            
            with open(input_path, 'r', encoding='utf-8') as f:
                lines = f.read().splitlines()
                
            for line in lines:
                line = line.strip()
                if len(line) > 10:
                    line = line.strip()
                    tokens = self.tokenizer(
                        line,
                        max_length = 256,
                        padding = 'max_length',
                        truncation = True,
                        return_tensors = 'pt' 
                    )
                    
                    input_ids.append(tokens['input_ids'].squeeze(0))
                    attention_mask.append(tokens['attention_mask'].squeeze(0))
                    
            tokenized_data = {
                'input_ids' : torch.stack(input_ids),
                'attention_mask' : torch.stack(attention_mask)
            }
                    
            torch.save(tokenized_data, output_path)
            print(f"saved {len(input_ids)} Language Modelling {split} Samples")
            
    # Tokenizing Chatbot Data
    def tokenize_chatbot_data(self):
        
        for split in ['train', 'valid', 'test']:
            
            input_path = f"NeuroFormer/data/raw/Chatbot Data/{split}.txt"
            output_path = f"NeuroFormer/data/tokenized/Chatbot Data/{split}.pt"
            input_ids = []
            attention_mask = []
            
            with open(input_path, 'r', encoding='utf-8') as f:
                conversations = f.read().strip().split('\n\n')
            
            for conv in conversations:
                if len(conv) > 20:
                    conv = conv.strip()
                    tokens = self.tokenizer(
                        conv,
                        max_length = 256,
                        padding = 'max_length',
                        truncation = True,
                        return_tensors = 'pt'
                    )
                    
                    input_ids.append(tokens['input_ids'].squeeze(0))
                    attention_mask.append(tokens['attention_mask'].squeeze(0))
                    
            tokenized_data = {
                'input_ids' : torch.stack(input_ids),
                'attention_mask' : torch.stack(attention_mask)
            }
            
            torch.save(tokenized_data, output_path)
            print(f"saved {len(input_ids)} Chatbot {split} Samples")
    
    def encode(self, text, add_special_tokens=True):
        return self.tokenizer(text, return_tensors='pt', add_special_tokens=add_special_tokens)

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True).strip().replace("Ġ", " ")
    
    def tokenize_all(self):
        print('Tokenization Started')
        self.tokenize_lm()
        self.tokenize_chatbot_data()
        print('Tokenization Done')
        
tokenizer = Tokenizer()
tokenizer.tokenize_all()

encoded_chat = tokenizer.encode("<user> Hello! <assistant> Hi there!")
decoded_chat = tokenizer.decode(encoded_chat['input_ids'][0])
print(f"Decoded Chatbot Output: {decoded_chat}")  