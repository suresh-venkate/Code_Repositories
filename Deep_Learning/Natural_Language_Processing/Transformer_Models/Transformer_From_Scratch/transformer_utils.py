##### Transformer Utilities

import numpy as np
import matplotlib.pyplot as plt
import math, copy, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

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
    """
    Arguments:
      x: Input tensor of shape [nb, nw], where 
         nb = batch size
         nw = number of words (positions) in the input
    Returns:
      Embedded tensor of shape [nb, nw, d_model]
    """
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
      max_posns: Maximum number of positions (words) that can be fed to this layer
      """
      super(PositionalEncoding, self).__init__()
      self.dropout = nn.Dropout(p = dropout)
      pe = torch.zeros(max_posns, d_model) # Placeholder to store pos. encodings
      posn_vec = torch.arange(0, max_posns).unsqueeze(1) # Position vector
      expn = torch.arange(0, d_model, 2) / d_model # Exponent term for the denominator
      pe[:, 0::2] = torch.sin(posn_vec / (10000 ** expn))
      pe[:, 1::2] = torch.cos(posn_vec / (10000 ** expn))
      pe = pe.unsqueeze(0) # Add batch axis
      self.register_buffer('pe', pe) # Register 'pe' as a non-model parameter
        
    def forward(self, x):
        """
        Arguments:
          x: Input tensor of shape [nb, nw, d_model], where 
             nb = batch size
             nw = number of words (positions) in the input
             d_model = size of input embeddings
        Returns:
            x: (Input tensor + positional embeddings) of shape [nb, nw, d_model]
        """        
        x = x + self.pe[:, :x.size(1)].clone().detach()
        return self.dropout(x)
    
### Class: ScaledDotProductAttention
class ScaledDotProductAttention(nn.Module):
  """
  Compute Scaled Dot Product Attention (Ref: Section 3.2.1, Figure 2 (left))
  """

  def __init__(self, scaling):
    """
    Arguments:
      scaling: scaling to use while computing attention. Set to sqrt(dk), where dk = 64 in the Transformer paper.
    """
    super(ScaledDotProductAttention, self).__init__()
    self.scaling = scaling
    self.softmax = nn.Softmax(dim = -1)
  
  def forward(self, Q, K, V, attn_mask = None, attn_dropout = None):
    """
    Arguments:
      Q: Query tensor of shape [nb, h, nw, dk], where 
         nb = batch size
         h = number of attention heads in the MHA (8 in the transformer paper)
         nw = number of words (or positions) in the input
         dk = dimension of the query vector (64 in the transformer paper)
      K: Key tensor of shape [nb, h, nw, dk]
      V: Value tensor of shape [nb, h, nw, dv], where
         dv = dimension of the value vector (64 in the transformer paper)
      attn_mask: Optional mask to mask out some query-key combinations
      attn_dropout: Dropout layer to add after softmax output of SDPA block
    Returns:
      probs: softmax(Q * K.T / scaling)
      attn: probs.V (Equation (1) in the Transformer paper)
    """
    scores = torch.matmul(Q, K.transpose(-2, -1)) # Matmul of Q and K - Computes Q(K.T). Shape = [nb, h, nw, nw]
    scores = scores / self.scaling # Apply Scaling to maintain original variance - Computes Q(K.T)/sqrt(dk). Shape = [nb, h, nw, nw]
    if attn_mask is not None: # Apply mask (optional)
      scores = scores.masked_fill(attn_mask == 0, -1e9)
    probs = self.softmax(scores) # Compute softmax. Shape = [nb, h, nw, nw]
    probs = attn_dropout(probs) # Apply dropout. Shape = [nb, h, nw, nw]
    attn = torch.matmul(probs, V) # Compute final attention output. Shape = [nb, h, nw, dv]

    return attn, probs

### Class: MultiHeadAttention
class MultiHeadAttention(nn.Module):
  """
  Compute Multi-Head Attention (Ref: Section 3.2.2, Figure 2 (right))
  """
  def __init__(self, h, d_model, attn_dropout = 0.1):
    """
    Arguments:
      h: Number of parallel attention layers (heads) [8 in the transformer paper]
      d_model: Size of input embeddings [512 in the transformer paper]
      attn_dropout: dropout value to use in MHA module
    """
    super(MultiHeadAttention, self).__init__()
    assert d_model % h == 0 # Confirm that input embedding size is a multiple of # heads
    self.d_k = d_model // h # Dimension of projected outputs for each head
    self.h = h # Number of heads
    self.attn = None # Placeholder to store attention softmax output

    # Define linear layers for projecting Q, K, V
    # This is a combined linear layer for all heads together
    self.wi_q = nn.Linear(d_model, d_model, bias = False)
    self.wi_k = nn.Linear(d_model, d_model, bias = False)
    self.wi_v = nn.Linear(d_model, d_model, bias = False)

    # Define SDPA instance
    self.sdpa = ScaledDotProductAttention(scaling = np.sqrt(self.d_k))

    # Define final FC and dropout layers
    self.fc = nn.Linear(d_model, d_model, bias = False)
    self.dropout = nn.Dropout(p = attn_dropout)
        
  def forward(self, Q, K, V, mask = None):
    """
    Arguments:
      Q: Query tensor of shape [nb, nw, d_model], where 
         nb = batch size
         nw = number of words (or positions) in the input
         d_model = dimension of the input embeddings (512 in the transformer paper)
      K: Key tensor of shape [nb, nw, d_model]
      V: Value tensor of shape [nb, nw, d_model], where
      attn_mask: Optional mask to mask out some query-key combinations
      attn_dropout: Dropout layer to add after softmax output of SDPA block
    Returns:
      x: output of the MHA layer of shape [nb, nw, d_model]
    """    
    if mask is not None:
      mask = mask.unsqueeze(1) # Head axis broadcasting - Same mask applied to all h heads.
    
    nb = Q.size(0) # Extract number of batches in Q 
    # Pass Q, K, V through linear layers and separate each head to get
    # outputs of shape nb x nw x h x dk
    Q = self.wi_q(Q).view(nb, -1, self.h, self.d_k)
    K = self.wi_k(K).view(nb, -1, self.h, self.d_k)
    V = self.wi_v(V).view(nb, -1, self.h, self.d_k)

    # Transpose nw and h for attention dot product
    # Shape is now nb x h x nw x dk
    Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)
        
    # Apply Scaled Dot Product Attention
    # Shape of x is nb x h x nw x dk
    x, self.attn = self.sdpa(Q, K, V, attn_mask = mask, attn_dropout = self.dropout)

    # Transpose h and nw dimensions and concatenate all heads together into 
    # a single unit of dimension nb x nw x d_model 
    x = x.transpose(1, 2).contiguous().view(nb, -1, self.h * self.d_k)

    # Apply final linear layer and dropout to return final MHA output of shape (nb x nw x d_model)
    x = self.dropout(self.fc(x))
    
    return x

### Class: PositionwiseFeedForward
class PositionwiseFeedForward(nn.Module):
  """
  Implements the position-wise feed-forward layer (Ref: Section 3.3, Eqn. (2))
  """
  def __init__(self, d_model, d_ff, pwff_dropout = 0.1):
    """
    Arguments:  
      d_model: Dimensionality of input and output of PWFFN layer (512 in Transformer Paper)
      d_ff: Dimension of hidden layer (2048 in Transformer Paper)
      pwff_dropout: Dropout value to use for position wise feedforward layers
    """
    super(PositionwiseFeedForward, self).__init__()
    self.fc_1 = nn.Linear(d_model, d_ff) # First transformation
    self.fc_2 = nn.Linear(d_ff, d_model) # Second transformation
    self.relu = nn.ReLU()  
    self.dropout = nn.Dropout(pwff_dropout)

  def forward(self, x):
    """
    Arguments:
        x: Input tensor of shape [nb, nw, d_model]
    """
    x = self.relu(self.fc_1(x)) # Intermediate output
    x = self.dropout(self.fc_2(x)) # Final output of shape [nb, nw, d_model]
    return x

### Class: AddAndNorm    
class AddAndNorm(nn.Module):
  """
  Add a residual connection followed by a layer norm. (Ref: Section 5.4, Residual Dropout)
  """
  def __init__(self, size):
    """
    Arguments:
        size: Size to use for LayerNorm layer. Will be set to d_model
    """
    super(AddAndNorm, self).__init__()
    self.norm = nn.LayerNorm(size, eps = 1e-6)
    #self.dropout = nn.Dropout(dropout)
    
  def forward(self, x, sublayer):
    """
    Arguments:
        x: Input signal to AddAndNorm Layer of shape [nb, nw, d_model]
        sublayer: Layer through which x is passed before doing Add-and-Norm
    """
    #return x + self.dropout(sublayer(self.norm(x)))
    return self.norm(x + sublayer((x)))

### Class: EncoderLayer 
class EncoderLayer(nn.Module):
  """
  Single Encoder Unit comprising of a multi-head-attention unit with add_and_norm followed by
  a position-wise-feed-forward-network with add_and_norm (Ref: Section 3.1, Encoder, Fig.1 left side)
  """
  def __init__(self, d_model, h, attn_dropout, d_ff, pwff_dropout):
    """
    Arguments:
      d_model: Size of input embeddings    
      h: Number of parallel attention layers (heads)
      attn_dropout: dropout value to use in MHA module
      d_ff: Dimension of hidden layer in position wise feedforward layer
      pwff_dropout: Dropout value to use for position wise feedforward layers
    """
    super(EncoderLayer, self).__init__()
    self.MHA_unit = MultiHeadAttention(h, d_model, attn_dropout)
    self.PWFFN = PositionwiseFeedForward(d_model, d_ff, pwff_dropout)
    self.addandnorm_MHA = AddAndNorm(d_model, attn_dropout)
    self.addandnorm_PWFFN = AddAndNorm(d_model, pwff_dropout)

  def forward(self, x, mask):
    x = self.addandnorm_MHA(x, lambda x: self.MHA_unit(x, x, x, mask))
    x = self.addandnorm_PWFFN(x, self.PWFFN)
    return x

### Class: Encoder
class Encoder(nn.Module):
  """
  Encoder is a stack of N EncoderLayers
  """
  def __init__(self, d_model, h, attn_dropout, d_ff, pwff_dropout, N):
    """
    Arguments:
      d_model: Size of input embeddings    
      h: Number of parallel attention layers (heads)
      attn_dropout: dropout value to use in MHA module
      d_ff: Dimension of hidden layer
      pwff_dropout: Dropout value to use for position wise feedforward layers    
      N: Number of EncoderLayers in the Encoder stack
    """
    super(Encoder, self).__init__()
    self.enclayer = EncoderLayer(d_model, h, attn_dropout, d_ff, pwff_dropout)
    self.enclayer_stack = clones(self.enclayer, N)
    self.norm = nn.LayerNorm(d_model, eps = 1e-6)
        
  def forward(self, x, mask):
    x = self.norm(x)
    for layer in self.enclayer_stack:
      x = layer(x, mask)
    return self.norm(x)

### Class: DecoderLayer
class DecoderLayer(nn.Module):
  """
  Single Decoder Unit comprising of two multi-head-attention units with add_and_norm followed by
  a position-wise-feed-forward-network with add_and_norm (Ref: Section 3.1, Decoder, Fig.1 right side)
  """
  def __init__(self, d_model, h, attn_dropout, d_ff, pwff_dropout):
    """
    Arguments:
      d_model: Size of input embeddings    
      h: Number of parallel attention layers (heads)
      attn_dropout: dropout value to use in MHA module
      d_ff: Dimension of hidden layer in position wise feedforward layer
      pwff_dropout: Dropout value to use for position wise feedforward layers
    """    
    super(DecoderLayer, self).__init__()
    #self.size = d_model
    self.self_MHA_unit = MultiHeadAttention(h, d_model, attn_dropout)
    self.src_MHA_unit = MultiHeadAttention(h, d_model, attn_dropout)
    self.PWFFN = PositionwiseFeedForward(d_model, d_ff, pwff_dropout)
    self.addandnorm_self_MHA = AddAndNorm(d_model, attn_dropout)
    self.addandnorm_src_MHA = AddAndNorm(d_model, attn_dropout)
    self.addandnorm_PWFFN = AddAndNorm(d_model, pwff_dropout)
 
  def forward(self, x, memory, src_mask, tgt_mask):
    m = memory
    x = self.addandnorm_self_MHA(x, lambda x: self.self_MHA_unit(x, x, x, tgt_mask))
    x = self.addandnorm_src_MHA(x, lambda x: self.src_MHA_unit(x, m, m, src_mask))      
    x = self.addandnorm_PWFFN(x, self.PWFFN)
    return x   

### Class: Decoder
class Decoder(nn.Module):
  """
  Decoder is a stack of N DecoderLayers
  """
  def __init__(self, d_model, h, attn_dropout, d_ff, pwff_dropout, N):
    """
    Arguments:
      d_model: Size of input embeddings    
      h: Number of parallel attention layers (heads)
      attn_dropout: dropout value to use in MHA module
      d_ff: Dimension of hidden layer
      pwff_dropout: Dropout value to use for position wise feedforward layers    
      N: Number of DecoderLayers in the Decoder stack
    """
    super(Decoder, self).__init__()
    self.declayer = DecoderLayer(d_model, h, attn_dropout, d_ff, pwff_dropout)
    self.declayer_stack = clones(self.declayer, N)
    self.norm = nn.LayerNorm(d_model, eps = 1e-6)    
        
  def forward(self, x, memory, src_mask, tgt_mask):
    x = self.norm(x)
    for layer in self.declayer_stack:
      x = layer(x, memory, src_mask, tgt_mask)
    return self.norm(x)

### Class: Generator        
class Generator(nn.Module):
  """
  Final output layer (linear + softmax) to be added after the decoder output
  """
  def __init__(self, d_model, vocab):
    """
    Arguments: 
      d_model: Size of the embedding vector
      vocab: Size of input vocabulary
    """
    super(Generator, self).__init__()
    self.proj = nn.Linear(d_model, vocab)
    self.smax = nn.LogSoftmax(dim = -1)

  def forward(self, x):
    x = self.proj(x)
    x = self.smax(x)
    return x

### Class: EncoderDecoder
class EncoderDecoder(nn.Module):
  """
  Overall Encoder + Decoder.
  """
  def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
    """
    Arguments:
        encoder: Encoder instance
        decoder: Decoder instance
        src_embed: Source embeddings (along with positional encoding) to pass to encoder 
        tgt_embed: Target embeddings (along with positional encoding) to pass to decoder
        generator: Output generator instance (Linear + Softmax)
    """
    super(EncoderDecoder, self).__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.src_embed = src_embed
    self.tgt_embed = tgt_embed
    self.generator = generator
    
  def encode(self, src, src_mask):
    return self.encoder(self.src_embed(src), src_mask)
      
  def decode(self, memory, src_mask, tgt, tgt_mask):
    return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
      
  def forward(self, src, tgt, src_mask, tgt_mask):
    "Take in and process masked src and target sequences."
    return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)
    
### Function: make_model
def make_model(src_vocab, tgt_vocab, N=6, 
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    position = PositionalEncoding(d_model, dropout)
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