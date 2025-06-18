# Importing Libraries
import torch
from utils import load_tokenizer

# A Simple Tokenizer Class that handles Decoder-Only, Encoder-Only, Encoder-Decoder tasks
class Tokenizer:
    
    def __init__(self):
        
        # Initializing all Tokenizers
        self.lm_tokenizer = load_tokenizer("lm")
        self.chat_tokenizer = load_tokenizer("chat")
        self.qa_tokenizer = load_tokenizer("qa")
        self.trans_tokenizer = load_tokenizer("trans")
        
        # Saving Modified Tokenizers
        self.lm_tokenizer.save_pretrained('NeuroFormer/tokenizers/lm')
        self.chat_tokenizer.save_pretrained('NeuroFormer/tokenizers/chat')
        self.qa_tokenizer.save_pretrained('NeuroFormer/tokenizers/qa')
        self.trans_tokenizer.save_pretrained('NeuroFormer/tokenizers/trans')
        
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
                    line = f"<s> {line} </s>"
                    tokens = self.lm_tokenizer(
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
    
    def encode_lm_input(self, text):
        return self.lm_tokenizer(
            text, 
            return_tensors='pt', 
            padding='max_length', 
            truncation=True, 
            max_length=256
        )

    def decode_lm_output(self, tokens):
        return self.lm_tokenizer.decode(tokens[0] if tokens.ndim > 1 else tokens, skip_special_tokens=True)
            
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
                    conv = f"<s> {conv.strip()} </s>"
                    tokens = self.chat_tokenizer(
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
    
    def encode_chatbot_input(self, text):
        return self.chat_tokenizer(
            text,
            return_tensors='pt',
            padding='max_length',
            truncation='longest_first',
            max_length=512
        )
    
    def decode_chatbot_output(self, tokens):
        return self.chat_tokenizer.decode(tokens[0] if tokens.ndim > 1 else tokens, skip_special_tokens=True)

    # Tokenizing Ques Ans Data
    def tokenize_qa_data(self):
        for split in ['train', 'valid', 'test']:
            input_path = f"NeuroFormer/data/raw/QuesAns Data/{split}.txt"
            output_path = f"NeuroFormer/data/tokenized/QuesAns Data/{split}.pt"
            input_ids = []
            attention_mask = []
            start_pos = []
            end_pos = []

            with open(input_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()

            examples = [ex for ex in text.split('\n\n') if ex.strip()]
            for ex in examples:
                context = question = answer = None
                for line in ex.split('\n'):
                    if line.startswith("<context>"):
                        context = line[len("<context>"):].strip()
                    elif line.startswith("<question>"):
                        question = line[len("<question>"):].strip()
                    elif line.startswith("<answer>"):
                        answer = line[len("<answer>"):].strip()
                if not all([context, question, answer]):
                    continue

                start_char = context.find(answer)
                if start_char == -1:
                    continue
                end_char = start_char + len(answer)

                encoding = self.qa_tokenizer(
                    context, question,
                    max_length=256, padding='max_length',
                    truncation='only_first',
                    return_offsets_mapping=True,
                    return_tensors='pt'
                )

                offsets = encoding['offset_mapping'].squeeze(0)

                start_idx = end_idx = None

                for idx, (s, e) in enumerate(offsets):
                    if s <= start_char < e and start_idx is None:
                        start_idx = idx
                    if s < end_char <= e and end_idx is None:
                        end_idx = idx

                if start_idx is not None and end_idx is None:
                    for idx in range(start_idx, len(offsets)):
                        if offsets[idx][1] >= end_char:
                            end_idx = idx
                            break

                if start_idx is None or end_idx is None or end_idx < start_idx:
                    continue 
                
                input_ids.append(encoding['input_ids'].squeeze(0))
                attention_mask.append(encoding['attention_mask'].squeeze(0))
                start_pos.append(start_idx)
                end_pos.append(end_idx)
                
            tokenized_data = {
                'input_ids' : torch.stack(input_ids),
                'attention_mask' : torch.stack(attention_mask),
                'start_idx': torch.tensor(start_pos, dtype=torch.long),
                'end_idx': torch.tensor(end_pos, dtype=torch.long)
            }

            torch.save(tokenized_data, output_path)
            print(f"Saved {len(input_ids)} QA {split} samples")

    
    def encode_qa_input(self, context, question):
        return self.qa_tokenizer(
            context,
            question,
            max_length=256,
            padding='max_length',
            truncation='only_first',
            return_tensors='pt'
        )

    def decode_qa_output(self, tokens):
        token_ids = tokens[0] if tokens.ndim > 1 else tokens
        token_list = self.qa_tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=True)
        return self.qa_tokenizer.convert_tokens_to_string(token_list).strip()
         
    # Tokenizing Translation Data
    def tokenize_translation_data(self):
        for split in ['train', 'valid', 'test']:
            input_path = f"NeuroFormer/data/raw/Translation Data/{split}.txt"
            output_path = f"NeuroFormer/data/tokenized/Translation Data/{split}.pt"
            input_ids = []
            attention_mask = []
            labels = []
            decoder_attention_mask= []

            with open(input_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()

            examples = [ex for ex in text.split('\n\n') if ex.strip()]
            for ex in examples:
                src = tgt = None
                for line in ex.split('\n'):
                    if line.startswith("<src>"):
                        src = line[len("<src>"):].strip()
                    elif line.startswith("<tgt>"):
                        tgt = line[len("<tgt>"):].strip()
                if src is None or tgt is None:
                    continue

                source_text = f"<source> {src}"
                target_text = f"<target> {tgt}"

                source_tokens = self.trans_tokenizer(
                    source_text,
                    max_length=256, padding='max_length',
                    truncation=True, return_tensors='pt'
                )
                target_tokens = self.trans_tokenizer(
                    target_text,
                    max_length=256, padding='max_length',
                    truncation=True, return_tensors='pt'
                )

                labels_tensor = target_tokens['input_ids'].squeeze(0)
                labels_tensor[labels_tensor == self.trans_tokenizer.pad_token_id] = -100
                labels_tensor = torch.clamp(labels_tensor, min=-100, max=self.trans_tokenizer.vocab_size - 1)
                
                input_ids.append(source_tokens['input_ids'].squeeze(0))
                attention_mask.append(source_tokens['attention_mask'].squeeze(0))
                labels.append(labels_tensor)
                decoder_attention_mask.append(target_tokens['attention_mask'].squeeze(0))
            
            tokenized_data = {
                'input_ids': torch.stack(input_ids),
                'attention_mask': torch.stack(attention_mask),
                'labels': torch.stack(labels),
                'decoder_attention_mask': torch.stack(decoder_attention_mask)
            }

            torch.save(tokenized_data, output_path)
            print(f"Saved {len(input_ids)} Translation {split} samples")

    
    def encode_translation_input(self, source: str) -> dict:
        return self.trans_tokenizer(
            f"<source> {source}",
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=256
        )

    def decode_translation_output(self, tokens: torch.Tensor) -> str:
        output = self.trans_tokenizer.decode(tokens[0] if tokens.ndim > 1 else tokens, skip_special_tokens=True)
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
decoded_chat = tokenizer.decode_chatbot_output(encoded_chat['input_ids'][0])
print(f"Decoded Chatbot Output: {decoded_chat}")  

encoded_qa = tokenizer.encode_qa_input("<context> AI is artificial intelligence.", "<question> What is AI?")
decoded_qa = tokenizer.decode_qa_output(encoded_qa['input_ids'][0])
print(f"Decoded QuesAns Output: {decoded_qa}")

encoded_trans = tokenizer.encode_translation_input("<source> How are you?")
decoded_trans = tokenizer.decode_translation_output(encoded_trans['input_ids'][0])
print(f"Decoded Translation Output: {decoded_trans}")