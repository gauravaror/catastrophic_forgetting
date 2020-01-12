import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from models.utils import Hardsigmoid, BernoulliST
from models.kv_memory import KeyValueMemory
# It's actually TransformerEncoder custom with PositionalEncoder but we use 
# name: TransformerRepresentation to avoid confusion with TransformerEncoder Representation.


class TransformerRepresentation(nn.Module):

    def __init__(self, emb_dim, nhead, nhid, nlayers, dropout=0.5,
                 use_memory=False, mem_size=None, mem_context_size=None,
                 inv_temp=None, use_binary=False):
        super(TransformerRepresentation, self).__init__()
        self.model_type = 'Transformer'
        self.emb_dim = emb_dim
        self.inv_temp = inv_temp
        self.memory = KeyValueMemory(use_memory=use_memory,
                                     emb_dim=self.emb_dim,
                                     mem_size=mem_size,
                                     mem_context_size=mem_context_size,
                                     inv_temp=self.inv_temp,
                                     use_binary=use_binary)
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(emb_dim, dropout)
        encoder_layers = TransformerEncoderLayer(self.memory.get_input_size(),
                                                 nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def add_target_pad(self):
        self.memory.add_target_pad()

    def get_output_dim(self):
        ## Transformer input size is same as output
        return self.memory.get_input_size()

    def forward(self, src, mask):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = src * math.sqrt(self.emb_dim)
        src = self.pos_encoder(src)
        src_input = self.memory(src)
        output = self.transformer_encoder(src_input, self.src_mask)
        return torch.mean(output, dim=1)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
