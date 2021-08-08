##### Transformer Utilities - Original

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable

from transformer_utils import clones, Embeddings, PositionalEncoding
from transformer_utils import Generator
from transformer_utils import PositionwiseFeedForward
from transformer_utils import ScaledDotProductAttention

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size, eps = 1e-6)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size, eps = 1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size, eps = 1e-6)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

### Class: MultiHeadAttention
class MultiHeadAttention(nn.Module):
  """
  Compute Multi-Head Attention (Ref: Section 3.2.2, Figure 2 (right))
  """
  def __init__(self, h, d_model, attn_dropout = 0.1):
    """
    Arguments:
      h: Number of parallel attention layers (heads)
      d_model: Size of input embeddings
      attn_dropout: dropout value to use in MHA module
    """
    super(MultiHeadAttention, self).__init__()
    assert d_model % h == 0 # Confirm that input embedding size is a multiple of # heads
    self.d_k = d_model // h # Dimension of projected outputs
    self.h = h # Number of heads
    self.attn = None # Placeholder to store attention softmax output

    # Define linear layers for projecting Q, K, V
    self.wi_q = nn.Linear(d_model, d_model, bias = False)
    self.wi_k = nn.Linear(d_model, d_model, bias = False)
    self.wi_v = nn.Linear(d_model, d_model, bias = False)

    # Define SDPA instance
    self.attention = ScaledDotProductAttention(scaling = self.d_k ** 0.5)

    # Define final FC and dropout layers
    self.fc = nn.Linear(d_model, d_model, bias = False)
    self.dropout = nn.Dropout(p = attn_dropout)
        
  def forward(self, Q, K, V, mask = None):
    if mask is not None:
      mask = mask.unsqueeze(1) # Head axis broadcasting - Same mask applied to all h heads.
    
    nb = Q.size(0) # Extract number of batches in Q 
    # Q, K, V are of shape nb x lq x d_model
    # Pass through linear layers and separate each head to get
    # outputs of shape nb x lq x h x d_k
    Q = self.wi_q(Q).view(nb, -1, self.h, self.d_k)
    K = self.wi_k(K).view(nb, -1, self.h, self.d_k)
    V = self.wi_v(V).view(nb, -1, self.h, self.d_k)

    # Transpose lq and h for attention dot product
    # Shape is now nb x h x lq x d_k
    Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)
        
    # Apply Scaled Dot Product Attention
    # Shape of x is nb x h x lq x d_k
    x, self.attn = self.attention(Q, K, V, attn_mask = mask)

    # Transpose nb and h dimensions and concatenate all heads together into 
    # a single unit of dimension h x d_k = d_model 
    x = x.transpose(1, 2).contiguous().view(nb, -1, self.h * self.d_k)

    # Apply final linear layer and dropout
    x = self.dropout(self.fc(x))
    
    return x

# class MultiHeadedAttention(nn.Module):
    # def __init__(self, h, d_model, dropout=0.1):
        # "Take in model size and number of heads."
        # super(MultiHeadedAttention, self).__init__()
        # assert d_model % h == 0
        # # We assume d_v always equals d_k
        # self.d_k = d_model // h
        # self.h = h
        # self.linears = clones(nn.Linear(d_model, d_model), 4)
        # self.attn = None
        # self.dropout = nn.Dropout(p=dropout)
        
        # # Define SDPA instance
        # self.attention = ScaledDotProductAttention(scaling = self.d_k ** 0.5)        
        
    # def forward(self, query, key, value, mask=None):
        # "Implements Figure 2"
        # if mask is not None:
            # # Same mask applied to all h heads.
            # mask = mask.unsqueeze(1)
        # nbatches = query.size(0)
        
        # # 1) Do all the linear projections in batch from d_model => h x d_k 
        # query, key, value =             [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             # for l, x in zip(self.linears, (query, key, value))]
        
        # # 2) Apply attention on all the projected vectors in batch. 
        
        # # x, self.attn = attention(query, key, value, mask=mask, 
                                 # # dropout=self.dropout)
        # x, self.attn = self.attention(query, key, value, attn_mask = mask, attn_dropout = self.dropout)                                 
        
        # # 3) "Concat" using a view and apply a final linear. 
        # x = x.transpose(1, 2).contiguous()              .view(nbatches, -1, self.h * self.d_k)
        # return self.linears[-1](x)

def make_model(src_vocab, tgt_vocab, N=6, 
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadAttention(h, d_model, attn_dropout = dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model

