# Importing Libraries
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import *

# Encoder Class of NeuroFormer used for QA 
class Encoder(nn.Module):

  def __init__(self, vocab_size, sequence_length, d_model, num_heads, hidden, num_layers, drop_prob=0.1, pad_token=0):
    super().__init__()

    self.pad_token = pad_token
    self.d_model = d_model

    # Embeddings
    self.embedding = nn.Embedding(vocab_size, d_model) # Input Embedding
    self.pos_enc = PositionalEncoding(sequence_length, d_model, drop_prob) # Positional Encoding

    self.layers = nn.ModuleList([
        EncoderBlock(d_model, num_heads, hidden, drop_prob)
        for _ in range(num_layers)
    ])

    self.qa_head = nn.Linear(d_model, 2) # For Logits Calculation

  def forward(self, input_ids, return_embeddings=False):

    # Embedding + Positional Encoding
    x = self.embedding(input_ids) * math.sqrt(self.d_model) # sqrt(d_model) is multiplied to Stabilize Variance as nn.Embedding generate number between 0 and 1
    x = self.pos_enc(x) # Applying Positional Encodings

    # Creating padding mask
    pad_mask_tensor = pad_mask(input_ids, self.pad_token)

    # Passing Through Multiple Encoder Layers
    for layer in self.layers:
      x = layer(x, pad_mask_tensor)

    if return_embeddings:
      return x
    else:
      logits = self.qa_head(x)
      start_logits = logits[:, :, 0]  # [B, T]
      end_logits = logits[:, :, 1]    # [B, T]
      return start_logits, end_logits