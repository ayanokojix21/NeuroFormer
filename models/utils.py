# Importing Libraries
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Setting Device to GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device

# Padding Mask
def pad_mask(seq, pad_token=0):

  # seq -> [batch_size, sequence_length]
  mask = (seq != pad_token).unsqueeze(1).unsqueeze(1) # [B, 1, 1, T]
  return mask

# Causal Mask
def causal_mask(sequence_length, device):

  # batch_size is referred as B and sequnece_length is referred as T while mentioning Dimentions

  # Creating a Lower Triangular matrix
  mask = torch.ones(sequence_length, sequence_length, device=device, dtype=torch.bool)
  mask = torch.tril(mask) # Dimentions: (T, T)

  # Adding batch_size, and num_heads
  mask = mask.unsqueeze(0).unsqueeze(0) # Dimentions: (1, 1, T, T)
  return mask

# Cross Mask
def cross_mask(src_seq, tgt_seq, src_pad_token=0, tgt_pad_token=0):

  # src_seq: [batch_size, src_len]
  # tgt_seq: [batch_size, tgt_len]

  batch_size, tgt_len = tgt_seq.shape
  batch_size, src_len = src_seq.shape

  # Create source padding mask: [batch_size, src_len]
  src_valid = (src_seq != src_pad_token)  # True for valid tokens

  # Create target padding mask: [batch_size, tgt_len]
  tgt_valid = (tgt_seq != tgt_pad_token)  # True for valid tokens

  # Each target position can attend to all valid source positions
  cross_mask = src_valid.unsqueeze(1).expand(-1, tgt_len, -1)  # [B, tgt_len, src_len]

  # Mask out padded target positions (they shouldn't attend to anything)
  tgt_mask = tgt_valid.unsqueeze(-1)  # [batch_size, tgt_len, 1]
  mask = cross_mask & tgt_mask  # [batch_size, tgt_len, src_len]

  # Add num_heads: [batch_size, 1, tgt_len, src_len]
  return mask.unsqueeze(1)

# Combining Masks
def combine_masks(*masks): # Takes Variable mask input

  # If no Mask is passed retruns None
  if not masks:
      return None

  # Take the first mask as starting point
  combined = masks[0]

  # Iterate over rest of masks and if mask is not None, it combines and return True only if all mask agrees on that position
  for mask in masks[1:]:
      if mask is not None:
          combined = combined & mask

  return combined

# Scaled Dot Product Attention
def scaled_dot_product_attention(q, k, v, mask = None):
  d_k = q.size()[-1] # head_dim
  dot_product = torch.matmul(q, k.transpose(-1, -2)) # q: [B, num_heads, T, head_dim], kᵀ: [B, num_heads, head_dim, T], q @ kᵀ: [B, num_heads, T, T]
  scaled = dot_product / math.sqrt(d_k) # [B, num_heads, T, T]

  if mask is not None:
    # Convert boolean mask to additive mask (-inf for False positions)
    scaled = scaled.masked_fill(~mask , float('-inf')) # After Broadcasting: [B, num_heads, T, T]
    
  scaled = scaled.clamp(min=-1e9, max=1e9)
  attention = F.softmax(scaled, dim=-1) # attention: [B, num_heads, T, T]
  values = torch.matmul(attention, v) # v: [B, num_heads, T, head_dim] , Values: [B, num_heads, T, head_dim]

  return attention, values

# MultiHead Attention
class MultiHeadAttention(nn.Module): # nn.Module so that it can inherit nn.Module functions and Becomes reusable and modular block

  def __init__(self, d_model, num_heads):
    super().__init__()
    self.d_model = d_model # Dimentions of Input Embeddings
    self.num_heads = num_heads # Number of heads in the Multi-Head Attention
    self.head_dim = d_model // num_heads # Dimentions of Embedding passed in each Head
    self.qkv_layer = nn.Linear(d_model, 3 * d_model) # Projects the input as Query, Key and Value
    self.linear_layer = nn.Linear(d_model, d_model) # After combining all the heads this layer concatenates result back to input embedding dimentions

  def forward(self, x, mask = None):
    batch_size, sequence_length, d_model = x.size() # Input X is of dimentions: [B, T, d_model]
    qkv = self.qkv_layer(x) # It makes the dimentions: [B, T, 3 * d_model]

    qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim) # 3 * d_model is converted to (num_heads, 3 * head_dim) as to process in each seperate head num_heads is included and 3 * head_dim as it contains q,k,v combined
    qkv = qkv.permute(0, 2, 1, 3) # Reshaping it to [B, num_heads, T, 3 * head_dim] for efficient attention computation

    q, k, v = qkv.chunk(3, dim=-1) # Splitting the last dimention for q,k,v and making it to [B, num_heads, T, head_dim]
    attention, values = scaled_dot_product_attention(q, k, v, mask) # Calculating the attention and values

    values = values.transpose(1, 2).contiguous() # Combines the output of all heads
    values = values.reshape(batch_size, sequence_length, d_model) # dimentions: [B, T, d_model]
    out = self.linear_layer(values) # Final transformation to combine the output of all heads into d_model: [B, T, d_model]

    return out

# Layer Normalization
class LayerNormalization(nn.Module):

  def __init__(self, d_model, eps=1e-5):
    super().__init__()
    self.eps = eps
    self.d_model = d_model
    self.gamma = nn.Parameter(torch.ones(d_model)) # Learnable Parameter
    self.beta = nn.Parameter(torch.zeros(d_model)) # Learnable Parameter

  def forward(self, x):

    mean = x.mean(dim=-1, keepdim=True) # Mean
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True) # Variance
    std = (var + self.eps).sqrt() # Standard Deviation : eps is added to avoid division by 0

    y = (x - mean) / std # Output
    out = self.gamma * y + self.beta # Applying Learnable Parameters
    return out

# Positionwise Feed Forward Network
class PositionWiseFeedForward(nn.Module):

  def __init__(self, d_model, hidden, drop_prob=0.1):
    super().__init__()
    self.linear1 = nn.Linear(d_model, hidden) # Hidden Layer converts dim from d_model to hidden
    self.linear2 = nn.Linear(hidden, d_model) # Hidden Layer converts dim from hidden to d_model
    self.relu = nn.ReLU() # ReLU Activation Function
    self.dropout = nn.Dropout(p=drop_prob) # Dropout Layer

  def forward(self, x):
    x = self.linear1(x)
    x = self.relu(x)
    x = self.dropout(x)
    x = self.linear2(x)
    return x

# MultiHead Cross Attention
class MultiHeadCrossAttention(nn.Module):

  def __init__(self, d_model, num_heads):
    super().__init__()
    self.d_model = d_model # Dimentions of Input Embedding
    self.num_heads = num_heads # Number of Heads in Multi-Head Cross Attention
    self.head_dim = d_model // num_heads # Dimention of Embedding Passed in each head
    self.kv_layer = nn.Linear(d_model, 2 * d_model) # Projects the Input Embedding as Key, Value from Encoder Output
    self.q_layer = nn.Linear(d_model, d_model) # Projects the Output Embedding as Query from Decoder Input
    self.linear_layer = nn.Linear(d_model, d_model) # Concatenates the results of all heads and returning back the result to d_model dimentions

  def forward(self, x, y, cross_mask = None):
    batch_size, tgt_length, d_model = y.size() # Decoder Input Y of dimentions: [B, T_tgt, d_model]
    src_length = x.size(1) # Encoder Output X

    kv = self.kv_layer(x) # It makes dimentions: [B, T_src, 2 * d_model] i.e Key and Value Combined from Encoder Output
    q = self.q_layer(y) # It makes dimentions: [B, T_tgt, d_model] i.e Query from Decoder Input

    kv = kv.reshape(batch_size, src_length, self.num_heads, 2 * self.head_dim) # 2 * d_model is converted to num_heads as to process in seperate heads and 2 * head_dim i.e combined dimentions of key and value vector
    q = q.reshape(batch_size, tgt_length, self.num_heads, self.head_dim) # d_model is converted to num_heads as to process in seperate heads and head_dim that represents the query vector dimentions

    kv = kv.permute(0, 2, 1, 3)  # For efficient computation, new dimentions: [B, num_heads, T_src, 2 * head_dim]
    q = q.permute(0, 2, 1, 3) # For efficient computation, new dimentions: [B, num_heads, T_tgt, head_dim]

    k, v = kv.chunk(2, dim=-1) # Splitting the last dimention of kv and making it's dimention as [B, num_heads, T_src, head_dim]
    attention, values = scaled_dot_product_attention(q, k, v, cross_mask) # Getting the Attention and values vector

    values = values.transpose(1, 2).contiguous() # Combining back all the heads
    values = values.reshape(batch_size, tgt_length, d_model) # Dimentions: [B, T_tgt, d_model]
    out = self.linear_layer(values) # Learnable Parameter and converts back to original shape

    return out

# Positional Encodings
class PositionalEncoding(nn.Module):

  def __init__(self, sequence_length, d_model, drop_prob=0.1):
    super().__init__()
    self.pos_embedding = nn.Embedding(sequence_length, d_model) # [T, d_model]
    self.dropout = nn.Dropout(p = drop_prob)

  def forward(self, x):

    batch_size, sequence_length, d_model = x.size() # [B, T, d_model]
    positions = torch.arange(sequence_length, device=x.device) # [T]
    positions = positions.unsqueeze(0) # [1, T]
    pos_emb = self.pos_embedding(positions) # [1, T, d_model]

    x = x + pos_emb # [B, T, d_model] Broadcast Addition
    x = self.dropout(x)
    return x

# Encoder Block
class EncoderBlock(nn.Module):

  def __init__(self, d_model, num_heads, hidden, drop_prob):
    super().__init__()

    # SubLayer 1 (MultiHeadAttention + LayerNorm)
    self.self_attention = MultiHeadAttention(d_model, num_heads)
    self.norm1 = LayerNormalization(d_model)
    self.dropout1 = nn.Dropout(p=drop_prob)

    # SubLayer 2 (FeedForwardNetwork + LayerNorm)
    self.ffn = PositionWiseFeedForward(d_model, hidden, drop_prob)
    self.norm2 = LayerNormalization(d_model)
    self.dropout2 = nn.Dropout(p=drop_prob)

  def forward(self, x, pad_mask = None):

    # Self-Attention with residual network
    attn_out = self.self_attention(x, pad_mask)
    x = self.norm1(x + self.dropout1(attn_out))

    # Feed Forward Network with residual network
    ffn_out = self.ffn(x)
    x = self.norm2(x + self.dropout2(ffn_out))

    return x

# Decoder Block
class DecoderBlock(nn.Module):

  def __init__(self, d_model, num_heads, hidden, drop_prob):
    super().__init__()

    # SubLayer 1 (MultiHeadAttention + LayerNorm)
    self.self_attention = MultiHeadAttention(d_model, num_heads)
    self.norm1 = LayerNormalization(d_model)
    self.dropout1 = nn.Dropout(p=drop_prob)

    # SubLayer2 (MultiHeadCrossAttention + LayerNorm)
    self.cross_attention = MultiHeadCrossAttention(d_model, num_heads)
    self.norm2 = LayerNormalization(d_model)
    self.dropout2 = nn.Dropout(p=drop_prob)

    # SubLayer 3 (FeedForwardNetwork + LayerNorm)
    self.ffn = PositionWiseFeedForward(d_model, hidden, drop_prob)
    self.norm3 = LayerNormalization(d_model)
    self.dropout3 = nn.Dropout(p=drop_prob)

  def forward(self, y, encoder_output=None, self_mask=None, cross_mask=None):

    # x: Encoder Output
    # y: Decoder Input

    # Masked-Self Attention with residual connections
    attn_out = self.self_attention(y, self_mask)
    y = self.norm1(y + self.dropout1(attn_out))

    # Cross Attention (only if Encoder output is provided i.e x)
    if encoder_output is not None:
      cross_attn_out = self.cross_attention(encoder_output, y, cross_mask)
      y = self.norm2(y + self.dropout2(cross_attn_out))

    # Feed Forward Network with residual connections
    ffn_out = self.ffn(y)
    y = self.norm3(y + self.dropout3(ffn_out))

    return y