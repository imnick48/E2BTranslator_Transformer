import torch
import math
import torch.nn.functional as F
from torch import nn

def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = k.shape[-1]
    score = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        score = score.masked_fill(mask == 0, float('-inf'))
    attention = F.softmax(score, dim=-1)
    context = torch.matmul(attention, v)
    return attention, context

def create_causal_mask(size):
    mask = torch.ones(size, size)
    mask = torch.triu(mask, diagonal=1).bool()
    return ~mask 

def create_padding_mask(seq, pad_idx=0):
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

class MultiHeadAttention(nn.Module):
  def __init__(self,d_model,num_heads):
    super().__init__()
    self.d_model=d_model
    self.num_heads=num_heads
    self.head_dim=d_model//num_heads
    self.w_q=nn.Linear(d_model,d_model)
    self.w_k=nn.Linear(d_model,d_model)
    self.w_v=nn.Linear(d_model,d_model)
    self.w_o=nn.Linear(d_model,d_model)

  def forward(self,q,k,v,mask=None):
    batch_size=q.shape[0]
    q=self.w_q(q).view(batch_size,-1,self.num_heads,self.head_dim).transpose(1,2)
    k=self.w_k(k).view(batch_size,-1,self.num_heads,self.head_dim).transpose(1,2)
    v=self.w_v(v).view(batch_size,-1,self.num_heads,self.head_dim).transpose(1,2)
    attention,context=scaled_dot_product_attention(q,k,v,mask)
    context=context.transpose(1,2).contiguous().view(batch_size,-1,self.num_heads*self.head_dim)
    out=self.w_o(context)
    return out

class FeedForward(nn.Module):
  def __init__(self,d_model,d_exp=4):
    super().__init__()
    hidSize=d_model*d_exp
    self.ff1=nn.Linear(d_model,hidSize)
    self.dp=nn.Dropout(0.1)
    self.ff2=nn.Linear(hidSize,d_model)
    self.relu=nn.ReLU()
  def forward(self,x):
    x=self.ff1(x)
    x=self.relu(x)
    x=self.dp(x)
    x=self.ff2(x)
    return x

class AddNorm(nn.Module):
  def __init__(self,d_model):
    super().__init__()
    self.layerNorm=nn.LayerNorm(d_model)
    self.dp=nn.Dropout(0.1)
  def forward(self,x,sublayer):
    return self.layerNorm(x + self.dp(sublayer))

class EncoderBlock(nn.Module):
  def __init__(self,d_model,num_heads,d_exp=4):
    super().__init__()
    self.MultiHeadAttention=MultiHeadAttention(d_model,num_heads)
    self.FeedForward=FeedForward(d_model,d_exp)
    self.AddNorm1=AddNorm(d_model)
    self.AddNorm2=AddNorm(d_model)
    self.dp1=nn.Dropout(0.1)
    self.dp2=nn.Dropout(0.1)
  def forward(self,x,src_mask=None):
    x_res=x
    x=self.MultiHeadAttention(x,x,x,src_mask)
    x=self.dp1(x)
    x=self.AddNorm1(x_res,x)
    x_res=x
    x=self.FeedForward(x)
    x=self.dp2(x)
    x=self.AddNorm2(x_res,x)
    return x

class Encoder(nn.Module):
  def __init__(self,d_model,num_heads,num_layers=6,d_exp=4):
    super().__init__()
    self.layer=nn.ModuleList([EncoderBlock(d_model,num_heads,d_exp) for i in range(num_layers)])
  def forward(self,x,src_mask=None):
    for layer in self.layer:
      x=layer(x,src_mask)
    return x

class DecoderBlock(nn.Module):
  def __init__(self,d_model,num_heads,d_exp=4):
    super().__init__()
    self.MaskMultiHeadAttention=MultiHeadAttention(d_model,num_heads)
    self.AddNorm1=AddNorm(d_model)
    self.CrossAttention=MultiHeadAttention(d_model,num_heads)
    self.AddNorm2=AddNorm(d_model)
    self.FeedForward=FeedForward(d_model,d_exp)
    self.AddNorm3=AddNorm(d_model)
    self.dp1=nn.Dropout(0.1)
    self.dp2=nn.Dropout(0.1)
    self.dp3=nn.Dropout(0.1)
  def forward(self,x,encoder_out,tgt_mask=None,src_mask=None):
    x_res=x
    x=self.MaskMultiHeadAttention(x,x,x,tgt_mask)
    x=self.dp1(x)
    x=self.AddNorm1(x_res,x)
    x_res=x
    x=self.CrossAttention(x,encoder_out,encoder_out,src_mask)
    x=self.dp2(x)
    x=self.AddNorm2(x_res,x)
    x_res=x
    x=self.FeedForward(x)
    x=self.dp3(x)
    x=self.AddNorm3(x_res,x)
    return x

class Decoder(nn.Module):
  def __init__(self, d_model, num_heads, num_layers, d_exp=4):
    super().__init__()
    self.layer=nn.ModuleList([DecoderBlock(d_model,num_heads) for i in range(num_layers)])
  def forward(self,x,encoder_out,tgt_mask=None,src_mask=None):
    for layer in self.layer:
      x=layer(x,encoder_out,tgt_mask,src_mask)
    return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)


    def forward(self, x):
        x = x + self.pe[:x.size(0), :].detach() 
        return x

class InputEmbedding(nn.Module):
  def __init__(self,d_model,vocab_size):
    super().__init__()
    self.embedding=nn.Embedding(vocab_size,d_model)
    self.pos_encoder=PositionalEncoding(d_model)
  def forward(self,x):
    x=self.embedding(x)
    x=self.pos_encoder(x)
    return x

class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, encoder_vocab_size, decoder_vocab_size, num_layers=6, d_exp=4, pad_idx=0):
        super().__init__()
        self.EncoderInput = InputEmbedding(d_model, encoder_vocab_size)
        self.DecoderInput = InputEmbedding(d_model, decoder_vocab_size)
        self.encoder = Encoder(d_model, num_heads, num_layers, d_exp)
        self.decoder = Decoder(d_model, num_heads, num_layers, d_exp)
        self.linear = nn.Linear(d_model, decoder_vocab_size)
        self.pad_idx = pad_idx  # Store the padding index
  
    def forward(self, src, trg):
        # Create masks
        src_padding_mask = create_padding_mask(src, self.pad_idx)
        seq_len = trg.size(1)
        causal_mask = create_causal_mask(seq_len).to(trg.device)
        
        # Embedding
        src = self.EncoderInput(src)
        trg = self.DecoderInput(trg)
        
        # Forward through encoder and decoder
        src = self.encoder(src, src_padding_mask)  # This requires updating Encoder.forward
        trg = self.decoder(trg, src, causal_mask, src_padding_mask)  # This requires updating Decoder.forward
        trg = self.linear(trg)
        return trg