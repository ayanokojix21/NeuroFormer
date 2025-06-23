import json
import torch.nn as nn

# Special Tokens
special_tokens = ['<user>', '<assistant>']

# Function to build vocabulary based on char level tokenization
def build_vocab(text):
    
    chars = sorted(list(set(text)))
    vocab = {ch: i for i, ch in enumerate(chars)}
    
    offset = len(vocab)
    for i, token in enumerate(special_tokens):
        vocab[token] = offset + i    
    
    ivocab = {i: ch for ch, i in vocab.items()}
    return vocab, ivocab

# Pretraining Train Data
with open('NeuroFormer/data/pretrain/train.txt', 'r', encoding='utf-8') as f:
    pretrain_train_data = f.read()

# Fine-tuning Train Data 
with open('NeuroFormer/data/finetune/train.txt', 'r', encoding='utf-8') as f:
    finetune_train_data = f.read()

data = pretrain_train_data + finetune_train_data # Combining both dataset for building vocabulary
char2idx, idx2char = build_vocab(data) 
vocab_size = len(char2idx) # Size of vocabulary
print(f'Vocab Size: {vocab_size}')

# Converting char to idx
def encode(text, vocab):
    tokens = []
    i = 0
    while i < len(text):
        matched = False
        for token in special_tokens:
            if text[i:i+len(token)] == token:
                tokens.append(vocab[token])
                i += len(token)
                matched = True
                break
        if not matched:
            tokens.append(vocab[text[i]])
            i += 1
    return tokens

# Converting idx to char
def decode(indices, ivocab):
    return ''.join([ivocab[i] for i in indices])

# Saving Tokenizers
with open('NeuroFormer/tokenizer/char2idx.json', 'w') as f:
    json.dump(char2idx, f, ensure_ascii=False)

with open('NeuroFormer/tokenizer/idx2char.json', 'w') as f:
    json.dump(idx2char, f, ensure_ascii=False)

# Building a Tokenizer Class for reuse during inference and training
class Tokenizer(nn.Module):
    
    def __init__(self, vocab):
        super().__init__()        
        self.char2idx = vocab
        self.idx2char = {i: ch for ch, i in vocab.items()}
        self.vocab_size = len(vocab)
        
    def encode(self, text):
        return encode(text, self.char2idx)
    
    def decode(self, ids):
        return ''.join([self.idx2char[i] for i in ids])