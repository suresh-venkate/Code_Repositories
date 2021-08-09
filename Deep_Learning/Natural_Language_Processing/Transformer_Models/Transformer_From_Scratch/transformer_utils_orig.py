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
from transformer_utils import ScaledDotProductAttention, MultiHeadAttention
from transformer_utils import AddAndNorm
from transformer_utils import EncoderLayer, Encoder
from transformer_utils import DecoderLayer, Decoder

# ### Class: DecoderLayer
# class DecoderLayer(nn.Module):
  # """
  # Single Decoder Unit comprising of two multi-head-attention units with add_and_norm followed by
  # a position-wise-feed-forward-network with add_and_norm (Ref: Section 3.1, Decoder, Fig.1 right side)
  # """
  # def __init__(self, d_model, h, attn_dropout, d_ff, pwff_dropout):
    # """
    # Arguments:
      # d_model: Size of input embeddings    
      # h: Number of parallel attention layers (heads)
      # attn_dropout: dropout value to use in MHA module
      # d_ff: Dimension of hidden layer in position wise feedforward layer
      # pwff_dropout: Dropout value to use for position wise feedforward layers
    # """    
    # super(DecoderLayer, self).__init__()
    # #self.size = d_model
    # self.self_MHA_unit = MultiHeadAttention(h, d_model, attn_dropout)
    # self.src_MHA_unit = MultiHeadAttention(h, d_model, attn_dropout)
    # self.PWFFN = PositionwiseFeedForward(d_model, d_ff, pwff_dropout)
    # self.addandnorm_self_MHA = AddAndNorm(d_model, attn_dropout)
    # self.addandnorm_src_MHA = AddAndNorm(d_model, attn_dropout)
    # self.addandnorm_PWFFN = AddAndNorm(d_model, pwff_dropout)
 
  # def forward(self, x, memory, src_mask, tgt_mask):
    # m = memory
    # x = self.addandnorm_self_MHA(x, lambda x: self.self_MHA_unit(x, x, x, tgt_mask))
    # x = self.addandnorm_src_MHA(x, lambda x: self.src_MHA_unit(x, m, m, src_mask))      
    # x = self.addandnorm_PWFFN(x, self.PWFFN)
    # return x  

# class DecoderLayer(nn.Module):
    # "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    # def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        # super(DecoderLayer, self).__init__()
        # self.size = size
        # self.self_attn = self_attn
        # self.src_attn = src_attn
        # self.feed_forward = feed_forward
        # self.sublayer = clones(AddAndNorm(size, dropout), 3)
 
    # def forward(self, x, memory, src_mask, tgt_mask):
        # "Follow Figure 1 (right) for connections."
        # m = memory
        # x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        # return self.sublayer[2](x, self.feed_forward)
        
# ### Class: Decoder
# class Decoder(nn.Module):
  # """
  # Decoder is a stack of N DecoderLayers
  # """
  # def __init__(self, d_model, h, attn_dropout, d_ff, pwff_dropout, N):
    # """
    # Arguments:
      # d_model: Size of input embeddings    
      # h: Number of parallel attention layers (heads)
      # attn_dropout: dropout value to use in MHA module
      # d_ff: Dimension of hidden layer
      # pwff_dropout: Dropout value to use for position wise feedforward layers    
      # N: Number of DecoderLayers in the Decoder stack
    # """
    # super(Decoder, self).__init__()
    # self.declayer = DecoderLayer(d_model, h, attn_dropout, d_ff, pwff_dropout)
    # self.declayer_stack = clones(self.declayer, N)
    # self.norm = nn.LayerNorm(d_model, eps = 1e-6)    
        
  # def forward(self, x, memory, src_mask, tgt_mask):
    # #x = self.norm(x)
    # for layer in self.declayer_stack:
      # x = layer(x, memory, src_mask, tgt_mask)
    # return self.norm(x)

# class Decoder(nn.Module):
    # "Generic N layer decoder with masking."
    # def __init__(self, layer, N):
        # super(Decoder, self).__init__()
        # self.layers = clones(layer, N)
        # self.norm = nn.LayerNorm(layer.size, eps = 1e-6)
        
    # def forward(self, x, memory, src_mask, tgt_mask):
        # for layer in self.layers:
            # x = layer(x, memory, src_mask, tgt_mask)
        # return self.norm(x)

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
    
def make_model(src_vocab, tgt_vocab, N=6, 
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadAttention(h, d_model, attn_dropout = dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    # model = EncoderDecoder(
        # Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        # Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             # c(ff), dropout), N),
        # nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        # nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        # Generator(d_model, tgt_vocab))        
    model = EncoderDecoder(
        Encoder(d_model, h, dropout, d_ff, dropout, N),
        Decoder(d_model, h, dropout, d_ff, dropout, N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model

