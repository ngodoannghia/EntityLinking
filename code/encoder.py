import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np


class ScaleDotProductAttention(nn.Module):
    def __init__(self, scale, dropout_rate):
        super(ScaleDotProductAttention, self).__init__()
        self.scale = scale
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
    ## Query: batch_size, n_heads, query_len, head_dim
    ## Key: batch_size, n_heads, key_len, head_dim
    ## Value: batch_size, n_head, value_len, head_dim
    ## sourec mask: batch_size, 1, 1, source_seq_len 
    ## target mask: batch_size, 1, target_seq_len, target_seq_len
    def forward(self, query, key, value, mask=None):
        score = torch.matmul(query, key.transpose(-2, -1)) # batch_size, n_heads, query_len, vakue_len
        score = score / self.scale
        if mask is not None:
            score = score.masked_fill(mask == False, float('-inf'))
        

        ## batch_size, n_heads, query_len, value_len
        attn_probs = F.softmax(score, dim=-1)

        output = torch.matmul(self.dropout(attn_probs), value)

        return output, attn_probs

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = self.d_v = d_model // n_heads  # head dim
        self.dropout_rate = dropout_rate

        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)

        self.attention = ScaleDotProductAttention(np.sqrt(self.d_k), dropout_rate)
    
    def split_heads(self, x):

        batch_size = x.size(0)
        x = x.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2) # batch_size, n_heads, seq_len, d_k

        ## x: batch_size, n_heads, seq_len, head_dim
        return x
    
    def group_heads(self, x):

        batch_size = x.size(0)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)

        return x

    def forward(self, query, key, value, mask=None):

        Q = self.split_heads(self.W_q(query))
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))

        if mask is not None:
            mask = mask.unsqueeze(1)
        
        # x: batch_size, n_heads, query_len, head_dim
        # attn: batch_size, n_heads, query_len, value_len
        x, attn = self.attention(Q, K, V, mask)
        
        # x: batch_size, query_len, d_model
        x = self.group_heads(x)

        x = self.W_o(x)


        # x: batch_size, query_len, d_model
        # attn: batch_size, n_heads, query_len, value_len
        return x, attn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout_rate=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff 
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)

        self.W_1 = nn.Linear(d_model, d_ff)
        self.W_2 = nn.Linear(d_ff, d_model)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        
        # x: batch_size, seq_len, d_model
        x = self.dropout(self.leakyrelu(self.W_1(x)))
        x = self.W_2(x)

        # x: batch_size, seq_len, d_model
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout_rate=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        self.max_len = max_len

        self.dropout = nn.Dropout(dropout_rate)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1) # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):

        x = x + self.pe[:x.size(0), :]
        s = self.dropout(x)

        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate 

        self.attn_layer = MultiHeadAttention(d_model, n_heads, dropout_rate)
        self.attn_layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.ff_layer = PositionwiseFeedForward(d_model, d_ff, dropout_rate)
        self.ff_layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask):
        x1, _ = self.attn_layer(x, x, x, mask) # batch_size, source_seq_len, d_model
        
        x = self.attn_layer_norm(x + self.dropout(x1)) # batch_size, source_seq_len, d_model
        
        x1 = self.ff_layer(x) # batch_size, source_seq_len, d_model

        x = self.ff_layer_norm(x + self.dropout(x1)) # batch_size, source_seq_len, d_model


        # x: (batch_size, source_seq_len, d_model)
        return x 

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, pad_idx, dropout_rate=0.1, output_encoder=256, max_length_seq_sence=50, max_len=5000):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.pad_idx = pad_idx
        self.dropout_rate = dropout_rate
        self.max_len = max_len
        self.output_encoder = output_encoder
        self.max_length_seq_sence = max_length_seq_sence

        self.tok_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_embedding = PositionalEncoding(d_model,dropout_rate, max_len)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout_rate)
            for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        
        self.linear1 = nn.Linear(d_model * max_length_seq_sence, 1024)
        self.linear2 = nn.Linear(1024, output_encoder)
        self.dropout = nn.Dropout(dropout_rate)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        mask = self.get_pad_mask(x, self.pad_idx)

        batch_size = x.shape[0]

        x = self.tok_embedding(x)
        x = self.pos_embedding(x)

        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.layer_norm(x)

        x = x.reshape(batch_size, -1)

        x = self.dropout(self.leakyrelu(self.linear1(x)))
        x = self.linear2(x)

        return x
    
    def get_pad_mask(self, x, pad_idx):

        x = (x != pad_idx).unsqueeze(-2)

        return x
    
    # def averge_embed(self, x, batch_size):
    
    #     return torch.mean(x, dim=1, keepdim=True).reshape(batch_size, -1)