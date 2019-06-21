import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=512, dropout=0.1):
        super().__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=16):
        super().__init__()
        self.d_model = d_model

        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model,
                         device='cuda' if torch.cuda.is_available() else 'cpu', requires_grad=False)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x


class EncoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads):
        super(EncoderLayer, self).__init__()
        self.attention = torch.nn.MultiheadAttention(model_dim, num_heads)
        self.ff = FeedForward(model_dim)
        self.norm1 = Norm(model_dim)
        self.norm2 = Norm(model_dim)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        context, attn_weights = self.attention(x, x, x)
        x1 = self.norm1(x + context)
        x2 = self.ff(x1)
        x2 = self.norm2(x1 + x2)
        return x2


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    def __init__(self, model_dim=256, N=2, num_heads=1):
        super().__init__()
        self.N = N
        self.pe = PositionalEncoder(model_dim)
        self.layers = get_clones(EncoderLayer(model_dim, num_heads), N)
        self.norm = Norm(model_dim)

    def forward(self, x):
        x = self.pe(x.permute(1, 0, 2))
        for i in range(self.N):
            x = self.layers[i](x)
        return self.norm(x).permute(1, 0, 2)
