from typing import Iterator, List, Dict
import torch
import torch.optim as optim
import numpy as np

from sklearn.metrics import matthews_corrcoef

from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.dataset_readers.stanford_sentiment_tree_bank import StanfordSentimentTreeBankDatasetReader
from allennlp.data.vocabulary import Vocabulary

from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.models import Model

from allennlp.training.metrics import CategoricalAccuracy, Average
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer

from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.util import move_optimizer_to_cuda
from models.classifier import MainClassifier


from collections import defaultdict

@Model.register("mean_classifier")
# This works exactly like Main Classifier and adds an layer of complexity
# i.e mostly during evaluation of this thing. It decides logit using mean of classifier using examplers provided.
class MeanClassifier(MainClassifier):
  def __init__(self, word_embeddings: TextFieldEmbedder, encoder: Seq2VecEncoder, vocab: Vocabulary) -> None:
    super().__init__(word_embeddings, encoder, vocab)
    self.examplers = defaultdict(None)
    self.encoder_representation = defaultdict(None)
    self.mean_representation = defaultdict(None)
    self.evaluate_using_mean = False
    self.adding_mean_representation = False

  def forward(self, tokens: Dict[str, torch.Tensor], label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
    if not (self.current_task in self.examplers):
      self.examplers[self.current_task] = defaultdict(list)
      self.encoder_representation[self.current_task] = defaultdict(list)
    output = super().forward(tokens, label)
    if self.training or self.adding_mean_representation:
      for i in range(label.shape[0]):
        self.examplers[self.current_task][label[i].item()].append(tokens['tokens'][i])
        self.encoder_representation[self.current_task][label[i].item()].append(output['encoder_output'][i])
    if self.evaluate_using_mean:
      label_list = []
      for en_o in output['encoder_output']:
        distance = []
        for s in self.mean_representation[self.current_task].values():
          distance.append(torch.dist(en_o,s,2))
        trch_distance = torch.stack(distance)
        current_one = trch_distance.argmax()
        this_label = list(self.mean_representation[self.current_task].keys())[current_one]
        y_onehot = torch.FloatTensor(1, len(list(self.mean_representation[self.current_task].keys())))

        # In your for loop
        y_onehot.zero_()
        y_onehot.scatter_(1, torch.tensor([[this_label]]), 1)

        label_list.append(y_onehot)
      labels_mean_class = torch.cat(label_list)
      output['logits'] = labels_mean_class
    return output

  def get_mean_prune_sampler(self):
    for task in self.encoder_representation.keys():
      labels = self.encoder_representation[task]
      if not (task in self.mean_representation):
        self.mean_representation[task] = defaultdict(torch.Tensor)
      for key in labels.keys():
        encoder_representation = labels[key]
        mixed_representation = torch.stack(encoder_representation)
        mean_current = torch.mean(mixed_representation, 0)
        self.mean_representation[task][key] = mean_current



