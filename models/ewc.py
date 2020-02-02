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

    def update_penalty(self, t:int, model: nn.Module, datasets: list, vocab):
        if t == 1:
            return
        self.t = t
        self.model = model
        self.vocab = vocab
        self._old_len_dataset = self._len_dataset
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        if datasets:
            self._len_dataset = len(datasets)
            self.dataset = datasets
            self._diag_fisher()

        self._means[t] = {}
        for n, p in deepcopy(self.params).items():
            self._means[self.t][n] = variable(p.data)

    def _diag_fisher(self):

        self.model.eval()
        self._precision_matrices[self.t] = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            self._precision_matrices[self.t][n] = variable(p.data)

        iterator = BucketIterator(batch_size=512, sorting_keys=[("tokens", "num_tokens")])

        for input in iterator(self.dataset, num_epochs=1):
            self.model.zero_grad()
            #input = variable(input)
          
            if torch.cuda.is_available(): 
                input = move_to_device(input, torch.cuda.current_device())
            output = self.model(input['tokens'], input['label'], task_id=self.t-1)
            #label = output.max(1)[1].view(-1)
            #loss = F.nll_loss(F.log_softmax(output, dim=1), label)
            loss = output['loss']
            loss.backward()

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
            for n, p in self.model.named_parameters():
                if n in self._precision_matrices[key]:
                  _loss = self._precision_matrices[key][n] * (p - self._means[key][n]) ** 2
                  #print("Loss", _loss, " precision ", self._precision_matrices[key][n], "  Diff  ", (p-self._means[key][n]))
                  loss += _loss.sum()
        print(loss)
        return loss
