import torch
import torch.nn as nn

# Taken from Annotated Transformer Blog post (https://nlp.seas.harvard.edu/2018/04/03/attention.html#attention)
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.features = features
        self.a_2 = []
        self.b_2 = []
        #nn.modules.container.ModuleList()
        self.eps = eps
        self.add_norm_param()

    def add_norm_param(self):
        self.a_2.append(nn.Parameter(torch.ones(self.features)))
        self.b_2.append(nn.Parameter(torch.zeros(self.features)))

    def forward(self, x, tid):
        if len(self.a_2) < tid:
            self.add_norm_param()
        b,s,d = x.shape
        sa = x.reshape(b*s,d)
        mean = sa.mean(0, keepdim=True).squeeze(0)
        std = sa.std(0, keepdim=True).squeeze(0)
        #print(self.a_2[tid-1].shape, tid, (x-mean).shape, std.shape)
        return self.a_2[tid-1] * (x - mean) / (std + self.eps) + self.b_2[tid-1]
