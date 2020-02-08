from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data
from allennlp.data.iterators import BucketIterator
from allennlp.nn.util import move_to_device

def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)


class EWC(object):
    def __init__(self, model: nn.Module):
        self.model = model
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = {}
        self._len_dataset = 1
        self._old_len_dataset = 1

    def update_penalty(self, t:int, model: nn.Module, len_data):
        #if t == 1:
        #    return
        self.t = t
        self.model = model
        self._len_dataset = len_data
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._diag_fisher()

        self._means[t] = {}
        for n, p in deepcopy(self.params).items():
            self._means[self.t][n] = variable(p.data)

    def _diag_fisher(self):

        if self.t not in self._precision_matrices:
            self._precision_matrices[self.t] = {}
            for n, p in deepcopy(self.params).items():
                p.data.zero_()
                self._precision_matrices[self.t][n] = variable(p.data)

        for n, p in self.model.named_parameters():
            #print("Moving params ", p.grad, n)
            if (not (p.grad is None)) and n in self._precision_matrices[self.t]:
                #print(" grad ", self._precision_matrices[self.t][n], self._len_dataset, self._old_len_dataset)
                self._precision_matrices[self.t][n].data += ((p.grad.data ** 2)/self._len_dataset)

    def penalty(self, t:int):
        loss = 0
        if t == 1:
            return loss
        for key in self._precision_matrices.keys():
            if key == t:
                continue
            for n, p in self.model.named_parameters():
                if n in self._precision_matrices[key]:
                  _loss = self._precision_matrices[key][n] * (p - self._means[key][n]) ** 2
                  #print("Loss", _loss, " precision ", self._precision_matrices[key][n], "  Diff  ", (p-self._means[key][n]))
                  loss += _loss.sum()
        return loss
