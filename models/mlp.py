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
               args) -> None:
        super(MLP, self).__init__()
        self.emb_dim = embedding_dim
        self.hdim = hidden_dimension
        self.layers = num_layers
        self.linears = nn.modules.container.ModuleList()
        self.layer2bottle = nn.modules.container.ModuleList()
        self.bottle2layer = nn.modules.container.ModuleList()
        self.bottle_neck_dim = 40
        self.activation = nn.Sigmoid()
        self.binarizer = BernoulliST
        self.args = args
        self.alpha = args.alpha
        self.use_binary = use_binary
        self.linears.append(nn.Linear(self.emb_dim, self.hdim))
        for i in range(self.layers-1):
            self.linears.append(nn.Linear(self.hdim, self.hdim))
        if self.use_binary:
            for i in range(self.layers):
                self.layer2bottle.append(nn.Linear(self.hdim, self.bottle_neck_dim))
                self.bottle2layer.append(nn.Linear(self.bottle_neck_dim, self.hdim))
        self.binary_loss = 0

    @overrides
    def get_input_dim(self) -> int:
        return self.emb_dim

    @overrides
    def get_output_dim(self) -> int:
        return self.hdim

    def get_binary_loss(self) -> float:
        return self.binary_loss

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor):
        if mask is not None:
            tokens = tokens * mask.unsqueeze(-1).float()
        x = tokens
        self.binary_loss = 0
        for idx,layer in enumerate(self.linears):
            x = layer(x)
            x = self.activation(x)
            if self.use_binary:
                bot = self.layer2bottle[idx](x.mean(dim=1).squeeze(1))
                bot = self.activation(bot)
                bot = self.bottle2layer[idx](bot)
                bot = self.activation(bot)
                loss_binary = torch.sum(0.1*torch.log(bot) + (1-0.1)*torch.log(1-bot))
                self.binary_loss += (-self.alpha*loss_binary)
                #print("Loss Binary ", -0.001*loss_binary, bot.shape, torch.log(bot))
                binary = self.binarizer(bot)
                binary = binary.unsqueeze(2)
                binary = binary.expand_as(x.transpose(1,2)).transpose(1,2)
                #print("Binary ", binary.numel(), binary.nonzero().size(0))
                x = x*binary
                #print("Binary ", x)
        output = x.mean(dim=1).squeeze(1)
        #print("Output ", output.numel(), output.nonzero().size(0), self.binary_loss)
        return output

