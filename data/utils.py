import torch.nn as nn

# Special Tokens
special_tokens = ['<user>', '<assistant>']

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