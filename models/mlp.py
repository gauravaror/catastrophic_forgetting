import torch
import torch.nn as nn
from torch.nn import Linear
from overrides import overrides
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder

class MLP(Seq2VecEncoder):

    def __init__(self,
               embedding_dim: int,
               hidden_dimension: int,
               num_layers: int) -> None:
        super(MLP, self).__init__()
        self.emb_dim = embedding_dim
        self.hdim = hidden_dimension
        self.layers = num_layers
        self.linears = nn.modules.container.ModuleList()
        self.activation = nn.ReLU()
        self.linears.append(nn.Linear(self.emb_dim, self.hdim))
        for i in range(self.layers-1):
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
        for layer in self.linears:
            x = layer(x)
            x = self.activation(x)
        return x.mean(dim=1).squeeze(1)

