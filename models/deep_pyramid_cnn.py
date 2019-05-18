from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from overrides import overrides
import torch
from torch.nn import Conv1d, Linear

from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.nn import Activation


@Seq2VecEncoder.register("deep_pyramid_cnn")
class DeepPyramidCNN(Seq2VecEncoder):
    def __init__(self,
                 embedding_dim: int,
                 num_filters: int,
                 num_layers: int,
                 ngram_filter_sizes: Tuple[int, ...] = (2, 3, 4, 5),  # pylint: disable=bad-whitespace
                 conv_layer_activation: Activation = None,
                 output_dim: Optional[int] = None) -> None:
        super(DeepPyramidCNN, self).__init__()
        self.channel_size = num_filters
        self._embedding_dim = embedding_dim
        self.conv_region_embedding = nn.Conv2d(1, self.channel_size, (3, embedding_dim), stride=1)
        self.conv3 = nn.Conv2d(self.channel_size, self.channel_size, (3, 1), stride=1)
        self.pooling = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding_conv = nn.ZeroPad2d((0, 0, 1, 1))
        self.padding_pool = nn.ZeroPad2d((0, 0, 0, 1))
        self.act_fun = nn.ReLU()
        self.linear_out = nn.Linear(2*self.channel_size, 2)
        self._output_dim = self.channel_size
        self.activations=[]


    @overrides
    def get_input_dim(self) -> int:
        return self._embedding_dim

    @overrides
    def get_output_dim(self) -> int:
        return self._output_dim

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor):  # pylint: disable=arguments-differ
        self.activations = []
        x = tokens
        our_shape = x.shape
        batch = x.shape[0]
        x = x.view(our_shape[0], 1, x.shape[1], x.shape[2])

        # Region embedding
        x = self.conv_region_embedding(x)        # [batch_size, channel_size, length, 1]

        x = self.padding_conv(x)                      # pad保证等长卷积，先通过激活函数再卷积
        x = self.act_fun(x)
        self.activations.append([x.data.clone().cpu()])
        x = self.conv3(x)
        x = self.padding_conv(x)
        x = self.act_fun(x)
        self.activations.append([x.data.clone().cpu()])
        x = self.conv3(x)

        while x.size()[-2] >= 2:
            x = self._block(x)
        x = x.view(batch, self._output_dim)
        #x = self.linear_out(x)
        return x,self.activations

    def _block(self, x):
        # Pooling
        x = self.padding_pool(x)
        px = self.pooling(x)

        # Convolution
        x = self.padding_conv(px)
        x = F.relu(x)
        self.activations.append([x.data.clone().cpu()])
        x = self.conv3(x)

        x = self.padding_conv(x)
        x = F.relu(x)
        self.activations.append([x.data.clone().cpu()])
        x = self.conv3(x)

        # Short Cut
        #x = x + px

        return x

    def predict(self, x):
        self.eval()
        out = self.forward(x)
        predict_labels = torch.max(out[0], 1)[1]
        self.train(mode=True)
        return predict_labels, self.activations
