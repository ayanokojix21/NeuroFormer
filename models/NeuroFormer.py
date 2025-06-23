# Importing Libraries
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import *
from models.encoder import Encoder
from models.decoder import Decoder
from models.encoder_decoder import EncoderDecoder

# NeuroFormer Wrapper Class for all models
class NeuroFormer(nn.Module):

  def __init__(self, mode, vocab_size, sequence_length, d_model, num_heads, hidden, num_layers, drop_prob=0.1, num_encoder_layers=None, num_decoder_layers=None):
    super().__init__()

    self.mode = mode

    if mode == 'encoder_only':
      self.model = Encoder(vocab_size, sequence_length, d_model, num_heads, hidden, num_layers, drop_prob)

    elif mode == 'decoder_only':
      self.model = Decoder(vocab_size, sequence_length, d_model, num_heads, hidden, num_layers, drop_prob)

    elif mode == 'encoder_decoder':
      enc_layers = num_encoder_layers or num_layers
      dec_layers = num_decoder_layers or num_layers
      self.model = EncoderDecoder(vocab_size, sequence_length, d_model, num_heads, hidden, enc_layers, dec_layers, drop_prob)

  def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)