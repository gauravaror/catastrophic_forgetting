import torch
import torch.nn as nn
from torch.nn import Linear
from overrides import overrides
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
import copy
from models.utils import Hardsigmoid, BernoulliST, RoundST

class MLP(Seq2VecEncoder):

    def __init__(self,
               embedding_dim: int,
               hidden_dimension: int,
               num_layers: int,
               alpha_sharp: float,
               use_binary: bool) -> None:
        super(MLP, self).__init__()
        self.emb_dim = embedding_dim
        self.hdim = hidden_dimension
        self.layers = num_layers
        self.linears = nn.modules.container.ModuleList()
        self.activation = nn.Sigmoid()
        self.loss_sharpening = nn.MSELoss(reduction='mean')
        self.alpha = alpha_sharp
        self.binarizer = BernoulliST
        self.use_binary = use_binary
        self.linears.append(nn.Linear(self.emb_dim, self.hdim))
        for i in range(self.layers-1):
            self.linears.append(nn.Linear(self.hdim, self.hdim))
        self.layers_output = []

    @overrides
    def get_input_dim(self) -> int:
        return self.emb_dim

    @overrides
    def get_output_dim(self) -> int:
        return self.hdim

    def get_sharpened_layer_loss(self, num_sharp: int , layer: int):
        sharp_layer = self.layers_output[layer-1]
        values, indexes = torch.topk(sharp_layer, num_sharp)
        sharp_mask = torch.zeros(sharp_layer.shape)
        unsharp_mask = torch.ones(sharp_layer.shape)
        if torch.cuda.is_available():
            sharp_mask = sharp_mask.cuda()
            unsharp_mask = unsharp_mask.cuda()
        sharp_mask[:,:,indexes] = 1
        unsharp_mask[:,:,indexes] = 0
        sharp_elem = sharp_layer*sharp_mask
        unsharp_elem = sharp_layer*unsharp_mask
        sharp_elem = sharp_elem + self.alpha*(1-sharp_elem)
        unsharp_elem = unsharp_elem - self.alpha*unsharp_elem
        sharpened = sharp_elem + unsharp_elem
        return self.loss_sharpening(sharp_layer, sharpened)

    def get_sharpened_loss(self, num_sharp: int):
        total_loss = 0
        for i in range(1, self.layers + 1):
            total_loss += self.get_sharpened_layer_loss(num_sharp, i)
        return total_loss

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor):
        if mask is not None:
            tokens = tokens * mask.unsqueeze(-1).float()
        x = tokens
        self.layers_output.clear()
        for layer in self.linears:
            x = layer(x)
            x = self.activation(x)
            self.layers_output.append(x)
            if self.use_binary:
                binary = self.binarizer(x)
                x = x*binary
                #print("Binary ", x)
        output = x.mean(dim=1).squeeze(1)
        return output

