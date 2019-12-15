# Code taken from here: https://github.com/nabihach/IDA/blob/master/nli/encoder.py , Paper : https://openreview.net/forum?id=rJgkE5HsnV
from typing import Iterator, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoder_IDA import EncoderRNN
import numpy as np
import torch

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
        return torch.softmax(inp@self.coef, inp.dim()-1)

class HashedMemoryRNN(EncoderRNN):

    def __init__(self, e_dim, h_dim, mem_size=500, inv_temp=1,
                 num_layers=1, dropout=0.0, base_rnn=nn.LSTM,
                 dropout_p=0.1, bidirectional=False,
                 batch_first=True, memmory_embed=None):
        super(HashedMemoryRNN, self).__init__(e_dim=e_dim, h_dim=h_dim, mem_size=mem_size,
                                              inv_temp=inv_temp, num_layers=num_layers,
                                              dropout=dropout, base_rnn=base_rnn,
                                              bidirectional=bidirectional, batch_first=batch_first)
        self.acc_slots = 20
        self.memory_embeddings = memmory_embed
        self.hh = Hash(self.memory_embeddings.get_output_dim(), self.mem_size)

    def access_memory(self, embedded : torch.Tensor, mem_v: nn.Linear):
        accessd_mem = self.hh.hash(embedded, self.mem_size)
        #print(accessd_mem, accessd_mem.shape, accessd_mem.sum(2)) # 1xBxembedding_dim
        memory_context = mem_v(accessd_mem)
        return memory_context
        
    def forward(self, input_seqs, input_lengths, hidden_f=None, hidden_b=None, mem_tokens=None):
        input_seqs = input_seqs.permute(1,0,2)
        mem_embeddings = self.memory_embeddings(mem_tokens)
        mem_embeddings = mem_embeddings.permute(1,0,2)
        #print(input_seqs.size(), input_lengths.size())
        max_input_length, batch_size,  _ = input_seqs.size()

        hidden_f_a = None
        hidden_b_a = None
        lengths2 = torch.LongTensor(batch_size)
        mask = sequence_mask(lengths2, max_len=max_input_length).transpose(0, 1).unsqueeze(2)

        #print(" masks ", input_lengths, input_lengths.size())
        #mask = input_lengths
        forward_outs = torch.Tensor(torch.zeros(max_input_length, batch_size, self.hidden_size))
        backward_outs = torch.Tensor(torch.zeros(max_input_length, batch_size, self.hidden_size))
        forward_hiddens = torch.Tensor(torch.zeros(max_input_length, batch_size, self.hidden_size))
        dummy_tensor = torch.Tensor(torch.zeros(1, batch_size, self.mem_context_size))

        if USE_CUDA:
            lengths2 = lengths2.cuda()
            mask = mask.cuda()
            forward_outs = forward_outs.cuda()
            backward_outs = backward_outs.cuda()
            forward_hiddens = forward_hiddens.cuda()
            dummy_tensor = dummy_tensor.cuda()

        #forward pass
        for i in range(max_input_length):
            embedded = input_seqs[i]
            mem_embedded = mem_embeddings[i]
            embedded = self.dropout(embedded)
            embedded = embedded.unsqueeze(0)  # S=1 x B x N
            mem_embedded = mem_embedded.unsqueeze(0)  # S=1 x B x N


            # Calculate attention from memory:
            ######################################
            if hidden_f is not None:
                memory_context = self.access_memory(mem_embedded, self.M_v_fwd[0])
           ######################################

                rnn_input = torch.cat((embedded, memory_context), 2)
                output_f, hidden_f_new = self.forward_rnn(rnn_input, hidden_f_a)
            else:
                rnn_input = torch.cat((embedded, dummy_tensor), 2)
                output_f, hidden_f_new = self.forward_rnn(rnn_input, hidden_f_a)

            output_f = torch.tanh(output_f)
            hidden_f_a = hidden_f_new
            hidden_f = hidden_f_new[0]
            #print("Hidden size", hidden_f_new[0].size())
            forward_hiddens[i,:,:] = hidden_f[-1,:,:]
            forward_outs[i] = output_f.squeeze(0)
            #print("Hidden size 2", hidden_f_new[0].size())

        #backward pass
        for i in range(1,max_input_length+1):
            embedded = input_seqs[-i]
            embedded = self.dropout(embedded)
            embedded = embedded.unsqueeze(0)  # S=1 x B x N


            # Calculate attention from memory:
            ######################################
            if hidden_b is not None:
                memory_context = self.access_memory(mem_embedded, self.M_v_bkwd[0])
            ######################################
                rnn_input = torch.cat( (embedded, memory_context), 2 )
                output_b, hidden_b_new = self.backward_rnn(rnn_input, hidden_b_a)
            else:
                rnn_input = torch.cat((embedded, dummy_tensor), 2)
                output_b, hidden_b_new = self.backward_rnn(rnn_input, hidden_b_a)

            output_b = torch.tanh(output_b)
            hidden_b_a = hidden_b_new
            hidden_b = hidden_b_new[0]
            #print("Back Mask unsquence", mask.size(), hidden_b.size())
            back_mask = mask[-i, :].unsqueeze(0).expand_as(hidden_b)
            hidden_b = hidden_b * back_mask.float()
            backward_outs[-i] = output_b.squeeze(0)

        masked_forward_outs = forward_outs * mask.float()
        masked_backward_outs = backward_outs * mask.float()
        masked_output = masked_forward_outs + masked_backward_outs #S x B x H
        
        hiddens_out = torch.Tensor(torch.zeros(2, batch_size, self.hidden_size))
        if USE_CUDA:
            hiddens_out = hiddens_out.cuda()
        for i in range(batch_size):
            hiddens_out[0,i,:] = forward_hiddens[input_lengths[i].sum()-1, i, :]
        
        hiddens_out[1,:,:] = hidden_b[-1,:,:]
        hiddens_out = hiddens_out.permute(1,2,0)
        hiddens_out = hiddens_out.contiguous().view(batch_size, -1)
        #return masked_forward_outs, masked_backward_outs, hiddens_out
        return hiddens_out, []
