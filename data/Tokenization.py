import torch
import json

# Function to build vocabulary based on char level tokenization
def build_vocab(text):
    
    chars = sorted(list(set(text)))
    vocab = {ch: i for i, ch in enumerate(chars)}
    ivocab = {i: ch for ch, i in vocab.items()}
    
    return vocab, ivocab

# Loading Training Data
with open('NeuroFormer/data/raw/train.txt', 'r', encoding='utf-8') as f:
    train_data = f.read()
    
# Loading Validation Data
with open('NeuroFormer/data/raw/valid.txt', 'r', encoding='utf-8') as f:
    valid_data = f.read()
    
data = train_data + valid_data # Combining both dataset for building vocabulary
char2idx, idx2char = build_vocab(data) 
vocab_size = len(char2idx) # Size of vocabulary
print(f'Vocab Size: {vocab_size}')

# Converting char to idx
def encode(text, vocab):
    return [vocab[c] for c in text]

# Converting idx to char
def decode(indices, ivocab):
    return ''.join([ivocab[i] for i in indices])

# Saving Tokenizers
with open('NeuroFormer/tokenizer/char2idx.json', 'w') as f:
    json.dump(char2idx, f)

with open('NeuroFormer/tokenizer/idx2char.json', 'w') as f:
    json.dump(idx2char, f)
    
# Saving tokenized data as .pt for faster loading
train_ids = torch.tensor(encode(train_data, char2idx), dtype=torch.long)
valid_ids = torch.tensor(encode(valid_data, char2idx), dtype=torch.long)

torch.save(train_ids, 'NeuroFormer/data/tokenized/train.pt')
torch.save(valid_ids, 'NeuroFormer/data/tokenized/valid.pt')