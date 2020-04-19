import torch
import torch.nn as nn
from torch.nn import Linear
from overrides import overrides
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from models.utils import Hardsigmoid, BernoulliST, RoundST

class MLP(Seq2VecEncoder):

    def __init__(self,
               embedding_dim: int,
               hidden_dimension: int,
               num_layers: int,
               use_binary: bool,
               batch_norm: bool) -> None:
        super(MLP, self).__init__()
        self.emb_dim = embedding_dim
        self.hdim = hidden_dimension
        self.layers = num_layers
        self.linears = nn.modules.container.ModuleList()
        if batch_norm:
            self.batch_norm = nn.modules.container.ModuleList()
        self.activation =  Hardsigmoid()
        self.binarizer = BernoulliST
        self.use_binary = use_binary
        self.linears.append(nn.Linear(self.emb_dim, self.hdim))
        if batch_norm:
            self.batch_norm.append(nn.BatchNorm1d(self.hdim))
        for i in range(self.layers-1):
            if batch_norm:
                self.batch_norm.append(nn.BatchNorm1d(num_features=self.hdim))
            self.linears.append(nn.Linear(self.hdim, self.hdim))

    @overrides
    def get_input_dim(self) -> int:
        return self.emb_dim

    @overrides
    def get_output_dim(self) -> int:
        return self.hdim

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor):
        if mask is not None:
            tokens = tokens * mask.unsqueeze(-1).float()
        x = tokens
        for idx,layer in enumerate(self.linears):
            x = layer(x)
            if hasattr(self, 'batch_norm'):
                x = x.permute(0,2,1)
                x = self.batch_norm[idx](x)
                x = x.permute(0,2,1)
            x = self.activation(x)
            if self.use_binary:
                binary = self.binarizer(x)
                x = x*binary
                #print("Binary ", x)
        output = x.mean(dim=1).squeeze(1)
        return output

