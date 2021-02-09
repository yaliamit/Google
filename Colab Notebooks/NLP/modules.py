import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from layers import *
from torch.autograd import Variable

class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=200, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
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
        pe = Variable(self.pe[:, :seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = x + pe
        return self.dropout(x)

class EncoderLayer(nn.Module):
        def __init__(self, d_model, heads, dropout=0.1):
            super().__init__()
            self.norm_1 = Norm(d_model)
            self.norm_2 = Norm(d_model)
            self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
            self.ff = FeedForward(d_model, dropout=dropout)
            self.dropout_1 = nn.Dropout(dropout)
            self.dropout_2 = nn.Dropout(dropout)

        def forward(self, x, mask):
            x2 = self.norm_1(x)
            x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
            x2 = self.norm_2(x)
            x = x + self.dropout_2(self.ff(x2))
            return x

# build a decoder layer with two multi-head attention layers and
# one feed-forward layer
class DecoderLayer(nn.Module):
        def __init__(self, d_model, heads, dropout=0.1):
            super().__init__()
            self.norm_1 = Norm(d_model)
            self.norm_2 = Norm(d_model)
            self.norm_3 = Norm(d_model)

            self.dropout_1 = nn.Dropout(dropout)
            self.dropout_2 = nn.Dropout(dropout)
            self.dropout_3 = nn.Dropout(dropout)

            self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
            self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
            self.ff = FeedForward(d_model, dropout=dropout)

        def forward(self, x, e_outputs, src_mask, trg_mask):
            x2 = self.norm_1(x)
            x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
            x2 = self.norm_2(x)
            x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, src_mask))
            x2 = self.norm_3(x)
            x = x + self.dropout_3(self.ff(x2))
            return x

