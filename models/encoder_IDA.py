# Code taken from here: https://github.com/nabihach/IDA/blob/master/nli/encoder.py , Paper : https://openreview.net/forum?id=rJgkE5HsnV
import torch
import torch.nn as nn
import torch.nn.functional as F


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



class EncoderRNN(nn.Module):

    def __init__(self, e_dim, h_dim, num_layers=1, dropout=0.0, base_rnn=nn.LSTM, dropout_p=0.1, bidirectional=False, batch_first=True):
        super(EncoderRNN, self).__init__()

        self.e_dim = e_dim
        self.hidden_size = h_dim
        self.n_layers = num_layers
        self.base_rnn = base_rnn
        self.dropout_p = dropout_p
        self.mem_context_size = 300
        
        self.dropout = nn.Dropout(dropout_p)
        self.forward_rnn = self.base_rnn(self.e_dim + self.mem_context_size, self.hidden_size, self.n_layers)
        self.backward_rnn = self.base_rnn(self.e_dim + self.mem_context_size, self.hidden_size, self.n_layers)

        self.M_k_fwd = nn.ModuleList()
        self.M_v_fwd = nn.ModuleList()
        self.M_k_bkwd = nn.ModuleList()
        self.M_v_bkwd = nn.ModuleList()
        self.add_target_pad(500)

    def get_output_dim(self):
        return 2*self.hidden_size
       
    def add_target_pad(self, mem_size):
        self.M_k_fwd.append(nn.Linear(self.hidden_size, mem_size, bias=False))
        self.M_v_fwd.append(nn.Linear(mem_size, self.mem_context_size, bias=False))
        self.M_k_bkwd.append(nn.Linear(self.hidden_size, mem_size, bias=False))
        self.M_v_bkwd.append(nn.Linear(mem_size, self.mem_context_size, bias=False))

        if USE_CUDA:
            self.M_k_fwd.cuda()
            self.M_v_fwd.cuda()
            self.M_k_bkwd.cuda()
            self.M_v_bkwd.cuda()

    def access_memory(self, hidden, mem_k, mem_v):
        key_representations = []
        for i,mem_key in enumerate(mem_k):
              key_representations.append(torch.exp(mem_key(hidden.squeeze(0))))
        alpha_tilda = torch.cat(key_representations)

        # Normalize the key representation like softmax, calculate the sum over all memory keys.
        alpha_sum = torch.sum(alpha_tilda, dim = 1).view(-1, 1) # batch x 1

        mem_context_arr = []
        for k,mem_val in zip(key_representations, mem_v):
            key_softmaxed = torch.div(k, alpha_sum.expand(k.size()))
            mem_context_arr.append(mem_val(key_softmaxed))
        mem_context = torch.stack(mem_context_arr, dim=0).sum(dim=0)
        if USE_CUDA:
            mem_context = mem_context.cuda()
        return mem_context

    def forward(self, input_seqs, input_lengths, hidden_f=None, hidden_b=None):
        input_seqs = input_seqs.permute(1,0,2)
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
            embedded = self.dropout(embedded)
            embedded = embedded.unsqueeze(0)  # S=1 x B x N


            # Calculate attention from memory:
            ######################################
            if hidden_f is not None:
                memory_context = self.access_memory(hidden_f, self.M_k_fwd, self.M_v_fwd)
           ######################################

                rnn_input = torch.cat( (embedded, memory_context.unsqueeze(0)), 2 )
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
                memory_context = self.access_memory(hidden_b, self.M_k_bkwd, self.M_v_bkwd)
            ######################################

                rnn_input = torch.cat( (embedded, memory_context.unsqueeze(0)), 2 )
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


    def init_hidden(self, batch_size):
        if self.base_rnn == nn.LSTM:
            return self._init_LSTM(batch_size)
        elif self.base_rnn == nn.GRU:
            return self._init_GRU(batch_size)
        else:
            raise NotImplementedError()

    def _init_GRU(self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))
        if USE_CUDA:
            return result.cuda()
        else:
            return result

    def _init_LSTM(self, batch_size):
        result = (torch.Tensor(torch.zeros(1, batch_size, self.hidden_size)), torch.Tensor(torch.zeros(1, batch_size, self.hidden_size)))
        if USE_CUDA:
            return result.cuda()
        else:
            return result
