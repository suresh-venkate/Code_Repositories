##### Transformer Utilities

import numpy as np
import matplotlib.pyplot as plt
import math, copy, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

### Class: Embeddings
class Embeddings(nn.Module):
  """
  Defines embedding layer to use at inputs of encoder and decoder. (Ref: Section 3.4)
  """
  def __init__(self, d_model, vocab):
    """
    Arguments:
      d_model: Size of the embedding vector
      vocab: Size of input vocabulary
    """
    super(Embeddings, self).__init__()
    self.emb = nn.Embedding(vocab, d_model) 
    self.d_model = d_model

  def forward(self, x):
    return self.emb(x) * math.sqrt(self.d_model)

### Class: PositionalEncoding
class PositionalEncoding(nn.Module):
    """
    Implement the PositionalEncoding Function. (Ref: Section 3.5)
    """
    def __init__(self, d_model, dropout, max_posns = 5000):
      """
      Arguments: d_model = Size of the embedding vector
      dropout: Dropout value to use at the output of this layer
      """
      super(PositionalEncoding, self).__init__()
      #self.norm = nn.LayerNorm(d_model, eps = 1e-6)
      self.dropout = nn.Dropout(p = dropout)
      pe = torch.zeros(max_posns, d_model) # Placeholder to store pos. encodings
      posn_vec = torch.arange(0, max_posns).unsqueeze(1) # Position vector
      expn = torch.arange(0, d_model, 2) / d_model # Exponent term for the denominator
      pe[:, 0::2] = torch.sin(posn_vec / (10000 ** expn))
      pe[:, 1::2] = torch.cos(posn_vec / (10000 ** expn))
      pe = pe.unsqueeze(0) # Add batch axis
      self.register_buffer('pe', pe) # Register 'pe' as a non-model parameter
        
    def forward(self, x):
      x = x + self.pe[:, :x.size(1)].clone().detach()
      return self.dropout(x)

### Class: AddAndNorm    
class AddAndNorm(nn.Module):
  """
  Add a residual connection followed by a layer norm. (Ref: Section 5.4, Residual Dropout)
  """
  def __init__(self, size):
    super(AddAndNorm, self).__init__()
    self.norm = nn.LayerNorm(size, eps = 1e-6)
    
  def forward(self, x, sublayer):
    return self.norm(x + sublayer(x))

### Function: clones
def clones(module, N):
  """
  Produce N identical layers of module
  Arguments:
    module: module to be cloned
    N: Number of time the module will be cloned
  Returns: Cloned module
  """
  cloned_module = nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
  return cloned_module
