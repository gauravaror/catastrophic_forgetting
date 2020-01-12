# Code taken from here: https://github.com/nabihach/IDA/blob/master/nli/encoder.py , Paper : https://openreview.net/forum?id=rJgkE5HsnV
from typing import Iterator, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoder_IDA import EncoderRNN
from models.kv_memory import KeyValueMemory
import numpy as np
import torch

from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper

USE_CUDA = torch.cuda.is_available()

def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = torch.LongTensor(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand

class Hash:
    def __init__(self, dimension: int, access_slots: int):
        self.p = 2147483647
        #self.p = 131071
        self.coef = torch.randint(1, self.p,(dimension, access_slots),dtype=torch.float, requires_grad=False)
        if USE_CUDA:
            self.coef = self.coef.cuda()

    def hash(self, inp, rang):
        return torch.Softmax(inp@self.coef, inp.dim()-1)

class HashedMemoryRNN(nn.Module):

    def __init__(self, e_dim, h_dim, num_layers=1,
                 dropout=0.0, base_rnn=nn.LSTM,
                 dropout_p=0.1, bidirectional=False,
                 batch_first=True, memmory_embed=None,
                 use_memory=False, mem_size=None, mem_context_size=None,
                 inv_temp=None, use_binary=False):
        super(HashedMemoryRNN, self).__init__()
        self.acc_slots = 10
        self.memory_embeddings = memmory_embed
        self.e_dim = e_dim
        self.hidden_size = h_dim
        #self.hh = [Hash(self.memory_embeddings.get_output_dim(), self.mem_size) for _ in range(self.acc_slots)]
        self.memory = KeyValueMemory(use_memory=use_memory,
                                     emb_dim=self.e_dim,
                                     mem_size=mem_size,
                                     mem_context_size=mem_context_size,
                                     inv_temp=inv_temp,
                                     use_binary=use_binary)
        self.lstm = PytorchSeq2VecWrapper(torch.nn.LSTM(self.memory.get_input_size(), h_dim,
                                              num_layers=num_layers,
                                              dropout=dropout,
                                              bidirectional=bidirectional,
                                              batch_first=batch_first))
        self.softmax = torch.nn.Softmax()
        if USE_CUDA:
            self.lstm = self.lstm.cuda()
            self.memory = self.memory.cuda()

    def get_output_dim(self):
        return self.hidden_size

    def add_target_pad(self):
        self.memory.add_target_pad()

    def forward(self, input_seqs, input_lengths, hidden_f=None, hidden_b=None, mem_tokens=None):
        max_input_length, batch_size,  _ = input_seqs.size()
        input_seq = self.memory(input_seqs)
        return self.lstm(input_seq, input_lengths), []
