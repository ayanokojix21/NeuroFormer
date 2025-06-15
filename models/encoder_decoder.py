# Importing Libraries
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import *

# EncoderDecoder Class of Neuroformer used for Machine Translation
class EncoderDecoder(nn.Module):

  def __init__(self, vocab_size, sequence_length, d_model, num_heads, hidden, num_encoder_layers, num_decoder_layers, drop_prob, pad_token=0):
    super().__init__()

    self.pad_token = pad_token
    self.d_model = d_model

    self.encoder_embedding = nn.Embedding(vocab_size, d_model)
    self.decoder_embedding = nn.Embedding(vocab_size, d_model)

    self.encoder_pos_enc = PositionalEncoding(sequence_length, d_model, drop_prob)
    self.decoder_pos_enc = PositionalEncoding(sequence_length, d_model, drop_prob)

    self.encoder_layers = nn.ModuleList([
        EncoderBlock(d_model, num_heads, hidden, drop_prob)
        for _ in range(num_encoder_layers)
    ])

    self.decoder_layers = nn.ModuleList([
        DecoderBlock(d_model, num_heads, hidden, drop_prob)
        for _ in range(num_decoder_layers)
    ])

    self.out = nn.Linear(d_model, vocab_size) # Final Linear Layer for Predicting Probablities

  def encode(self, src_ids):

    # Creating Padding mask
    src_pad_mask = pad_mask(src_ids, self.pad_token)

    # Embedding + Positional Encoding
    x = self.encoder_embedding(src_ids) * math.sqrt(self.d_model) # sqrt(d_model) is multiplied to Stabilize Variance as nn.Embedding generate number between 0 and 1
    x = self.encoder_pos_enc(x) # Applying Positional Encodings

    # Passing Through Multiple Encoder Layers
    for layer in self.encoder_layers:
      x = layer(x, src_pad_mask)

    return x

  def decode(self, tgt_ids, encoder_output, src_ids):

    tgt_len = tgt_ids.size(1)

    # Decoder self-attention mask (causal + padding)
    tgt_pad_mask = pad_mask(tgt_ids, self.pad_token)
    causal = causal_mask(tgt_len, tgt_ids.device)

    tgt_padding_expanded = tgt_pad_mask.expand(-1, -1, tgt_len, -1)
    tgt_padding_self_attn = tgt_padding_expanded & tgt_padding_expanded.transpose(-1, -2)
    self_mask = combine_masks(causal, tgt_padding_self_attn)

    # Cross Attention Mask
    cross_attn_mask = cross_mask(src_ids, tgt_ids, self.pad_token, self.pad_token)

    # Embedding + Positional Encoding
    y = self.decoder_embedding(tgt_ids) * math.sqrt(self.d_model) # sqrt(d_model) is multiplied to Stabilize Variance as nn.Embedding generate number between 0 and 1
    y = self.decoder_pos_enc(y) # Applying Positional Encodings

    for layer in self.decoder_layers:
      y = layer(y, encoder_output, self_mask=self_mask, cross_mask=cross_attn_mask)

    logits = self.out(y)
    return logits # [batch_size, tgt_len, vocab_size]

  def forward(self, src_ids, tgt_ids):

    # Encoder Block
    encoder_output = self.encode(src_ids)

    # Decoder Block
    logits = self.decode(tgt_ids, encoder_output, src_ids)
    return logits