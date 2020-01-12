import torch
import torch.nn as nn
from models.utils import Hardsigmoid, BernoulliST, RoundST

USE_CUDA = torch.cuda.is_available()

class KeyValueMemory(nn.Module):

    def __init__(self, use_memory=False, emb_dim=128, mem_size=None,
                 mem_context_size=None, bidirectional=False,
                 inv_temp=None, use_binary=False):
        super(KeyValueMemory, self).__init__()
        self.use_memory = use_memory
        self.inv_temp = inv_temp
        self.use_binary = use_binary

        if self.use_memory and (mem_size == None or mem_context_size == None):
            raise Exception("Use of memory is enabled and mem_size and mem_context are not passed")

        self.emb_dim = emb_dim
        self.mem_size = mem_size
        self.mem_context_size = mem_context_size
        self.bidirectional = bidirectional
        self.M_k_fwd = nn.ModuleList()
        self.M_v_fwd = nn.ModuleList()
        if self.bidirectional:
            self.M_k_bkwd = nn.ModuleList()
            self.M_v_bkwd = nn.ModuleList()

        self.add_target_pad()

        if self.use_binary:
            self.sigmoid = Hardsigmoid()
            self.binarizer = RoundST

    def add_target_pad(self):
        if self.use_memory:
            self.M_k_fwd.append(nn.Linear(self.emb_dim, self.mem_size, bias=False))
            self.M_v_fwd.append(nn.Linear(self.mem_size, self.mem_context_size, bias=False))
            if self.bidirectional:
                self.M_k_bkwd.append(nn.Linear(self.emb_dim, self.mem_size, bias=False))
                self.M_v_bkwd.append(nn.Linear(self.mem_size, self.mem_context_size, bias=False))

            if USE_CUDA:
                self.M_k_fwd.cuda()
                self.M_v_fwd.cuda()
                if self.bidirectional:
                    self.M_k_bkwd.cuda()
                    self.M_v_bkwd.cuda()

    def access_memory(self, hidden, mem_k, mem_v):
        key_representations = []
        for i,mem_key in enumerate(mem_k):
              mem_r = mem_key(hidden.squeeze(0))
              if self.inv_temp:
                  mem_r = self.inv_temp*mem_r
              key_representations.append(torch.exp(mem_r))
        alpha_tilda = torch.cat(key_representations, dim=hidden.dim()-1)

        # Normalize the key representation like softmax, calculate the sum over all memory keys.
        alpha_sum = torch.sum(alpha_tilda, dim = hidden.dim()-1)
        alpha_sum = alpha_sum.unsqueeze(hidden.dim()-1)

        mem_context_arr = []
        for k,mem_val in zip(key_representations, mem_v):
            key_softmaxed = torch.div(k, alpha_sum.expand(k.size()))
            if self.use_binary:
                key_softmaxed = self.binarizer(self.sigmoid(key_softmaxed))
            mem_context_arr.append(mem_val(key_softmaxed))
        mem_context = torch.stack(mem_context_arr).sum(dim=0)
        if USE_CUDA:
            mem_context = mem_context.cuda()
        return mem_context

    def get_input_size(self):
        if self.use_memory:
            return self.emb_dim + self.mem_context_size
        else:
            return self.emb_dim

    def forward(self, input_embedding, fwd=True):
        if not self.use_memory:
            return input_embedding
        if fwd:
            memory_context =  self.access_memory(input_embedding, self.M_k_fwd, self.M_v_fwd)
            src_input = torch.cat((input_embedding, memory_context), input_embedding.dim()-1)
            return src_input
        elif self.bidirectional:
            memory_context = self.access_memory(input_embedding, self.M_k_bkwd, self.M_v_bkwd)
            src_input = torch.cat((input_embedding, memory_context), input_embedding.dim()-1)
            return src_input
        else:
            raise Exception("Called with backward when memory is not intalized for bidirectional")
