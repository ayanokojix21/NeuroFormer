# Importing Libraries
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import *

# Decoder Class of NeuroFormer used for Language Modelling or Chat
class Decoder(nn.Module):

  def __init__(self, vocab_size, sequence_length, d_model, num_heads, hidden, num_layers, drop_prob=0.1, pad_token=0):
    super().__init__()

    self.pad_token = pad_token
    self.d_model = d_model

    # Embeddings
    self.embedding = nn.Embedding(vocab_size, d_model) # Input Embedding
    nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
    self.pos_enc = PositionalEncoding(sequence_length, d_model, drop_prob) # Positional Encoding

    self.layers = nn.ModuleList([
        DecoderBlock(d_model, num_heads, hidden, drop_prob)
        for _ in range(num_layers)
    ])

    self.out = nn.Linear(d_model, vocab_size) # Final Linear Layer for Predicting Probablities

  def forward(self, input_ids):

    seq_len = input_ids.size(1)

    # Creating mask
    causal = causal_mask(seq_len, input_ids.device)
    mask = causal

    # Embedding + Positional Encoding
    y = self.embedding(input_ids) * math.sqrt(self.d_model) # sqrt(d_model) is multiplied to Stabilize Variance as nn.Embedding generate number between 0 and 1
    y = self.pos_enc(y) # Applying Positional Encodings

    # Passing Through Multiple Decoder Layers (No Cross Attention)
    for layer in self.layers:
      y = layer(y, encoder_output=None, self_mask=mask)

    logits = self.out(y)
    return logits