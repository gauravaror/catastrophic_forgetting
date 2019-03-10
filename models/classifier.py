from typing import Iterator, List, Dict
import torch
import torch.optim as optim
import numpy as np

from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.dataset_readers.stanford_sentiment_tree_bank import StanfordSentimentTreeBankDatasetReader
from allennlp.data.vocabulary import Vocabulary

from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.models import Model

from allennlp.training.metrics import CategoricalAccuracy
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer

from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.util import move_optimizer_to_cuda

@Model.register("main_classifier")
class MainClassifier(Model):
  def __init__(self, word_embeddings: TextFieldEmbedder, encoder: Seq2VecEncoder, vocab: Vocabulary) -> None:
    super().__init__(vocab)
    self.word_embeddings = word_embeddings
    self.encoder = encoder
    self.vocab = vocab
    self.tasks_vocabulary = {"default": vocab}
    self.current_task = "default"
    self.num_task = 0
    self.classification_layers = torch.nn.ModuleList([torch.nn.Linear(in_features=self.encoder.get_output_dim(), out_features=self.vocab.get_vocab_size('labels'))])
    self.task2id = { "default": 0 }
    self.hidden2tag = self.classification_layers[self.task2id["default"]]
    self.accuracy = CategoricalAccuracy()
    self.loss_function = torch.nn.CrossEntropyLoss()

  def add_task(self, task_tag: str, vocab: Vocabulary):
    self.classification_layers.append(torch.nn.Linear(in_features=self.encoder.get_output_dim(), out_features=vocab.get_vocab_size('labels')))
    self.num_task = self.num_task + 1
    self.task2id[task_tag] = self.num_task
    self.tasks_vocabulary[task_tag] = vocab

  def set_task(self, task_tag: str):
    self.hidden2tag = self.classification_layers[self.task2id[task_tag]]
    self.current_task = task_tag
    self.vocab = self.tasks_vocabulary[task_tag]

  def forward(self, tokens: Dict[str, torch.Tensor], label: torch.Tensor = None) -> Dict[str, torch.Tensor]:

    mask = get_text_field_mask(tokens)
    embeddings = self.word_embeddings(tokens)
    encoder_out = self.encoder(embeddings, mask)
    tag_logits = self.hidden2tag(encoder_out)
    output = {'logits': tag_logits }

    if label is not None:
      self.accuracy(tag_logits, label)
      output["loss"] = self.loss_function(tag_logits, label)

    return output

  def get_metrics(self, reset: bool = False) -> Dict[str, float]:
    return {"accuracy": self.accuracy.get_metric(reset)}
