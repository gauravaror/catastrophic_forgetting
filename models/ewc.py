from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data
from allennlp.data.iterators import BucketIterator


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
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            self._precision_matrices[n] = variable(p.data)
        self._len_dataset = 1
        self._old_len_dataset = 1

    def update_penalty(self, t:int, model: nn.Module, datasets: list, vocab):
        if t == 1:
            return
        self.model = model
        self.vocab = vocab
        self._old_len_dataset = self._len_dataset
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        if datasets:
            self._len_dataset += len(datasets)
            self.dataset = datasets
            self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            self._means[n] = variable(p.data)

    def _diag_fisher(self):

        self.model.eval()

        iterator = BucketIterator(batch_size=512, sorting_keys=[("tokens", "num_tokens")])

        for input in iterator(self.dataset, num_epochs=1):
            self.model.zero_grad()
            #input = variable(input)
            output = self.model(input['tokens'], input['label'])
            #label = output.max(1)[1].view(-1)
            #loss = F.nll_loss(F.log_softmax(output, dim=1), label)
            loss = output['loss']
            loss.backward()

            for n, p in self.model.named_parameters():
                if not p.grad is None and n in self._precision_matrices:
                    self._precision_matrices[n].data += p.grad.data ** 2
                    self._precision_matrices[n].data = self._precision_matrices[n].data*(self._len_dataset/self._old_len_dataset)

    def penalty(self, t:int):
        loss = 0
        if t == 1:
            return loss
        for n, p in self.model.named_parameters():
            if n in self._precision_matrices:
              _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
              loss += _loss.sum()
        return loss
