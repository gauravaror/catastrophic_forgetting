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
from models.hashedIDA import HashedMemoryRNN

from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.util import move_optimizer_to_cuda

@Model.register("majority_class_classifier")
class MajorityClassifier(Model):
  def __init__(self, vocab: Vocabulary) -> None:
    super().__init__(vocab)
    self.vocab = vocab
    self.current_task = "default"
    self.tasks_vocabulary = {"default": vocab}
    self.classification_layers = torch.nn.ModuleList([torch.nn.Linear(in_features=10, out_features=self.vocab.get_vocab_size('labels'))])
    self.loss_function = torch.nn.CrossEntropyLoss()
    self.accuracy = CategoricalAccuracy()
    self.average  = Average()

  def add_task(self, task_tag: str, vocab: Vocabulary):
    self.tasks_vocabulary[task_tag] = vocab

  def set_task(self, task_tag: str):
    self.current_task = task_tag
    self.vocab = self.tasks_vocabulary[task_tag]

  def forward(self, tokens: Dict[str, torch.Tensor], label: torch.Tensor=None) -> Dict[str, torch.Tensor]:
   logi=np.zeros(2)
   if self.current_task == "cola":
     logi=np.zeros(2)
     logi[self.vocab.get_token_index("1", "labels")] = 1
     logi=torch.Tensor(np.repeat([logi], label.size(),0))
   elif self.current_task == "trec":
     logi=np.zeros(6)
     logi[self.vocab.get_token_index("ENTY", "labels")] = 1
     logi=torch.Tensor(np.repeat([logi], label.size(),0))
   elif self.current_task == "sst":
     logi=np.zeros(5)
     logi[self.vocab.get_token_index("3", "labels")] = 1
     logi=torch.Tensor(np.repeat([logi], label.size(),0))
   elif self.current_task == "subjectivity":
     logi=np.zeros(2)
     logi[self.vocab.get_token_index("SUBJECTIVE", "labels")] = 1
     logi=torch.Tensor(np.repeat([logi], label.size(0),0))
   output = {}
   #print("Going foward , do we have labels", label)
   if label is not None:
     _, preds = logi.max(dim=1)
     print(label, preds, matthews_corrcoef(label.data.cpu().numpy(), preds.data.cpu().numpy()))
     self.average(matthews_corrcoef(label.data.cpu().numpy(), preds.data.cpu().numpy()))
     print("mathew majority ", self.average.get_metric())
     self.accuracy(logi, label)
     output["loss"] = torch.tensor([0])
   return output

  def get_metrics(self, reset: bool = False) -> Dict[str, float]:
    return {"accuracy": self.accuracy.get_metric(reset), "average": self.average.get_metric(reset)}

  
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
    self.accuracy = CategoricalAccuracy()
    self.loss_function = torch.nn.CrossEntropyLoss()
    self.average  = Average()
    self.activations = []
    self.labels = []

  def add_task(self, task_tag: str, vocab: Vocabulary):
    self.classification_layers.append(torch.nn.Linear(in_features=self.encoder.get_output_dim(), out_features=vocab.get_vocab_size('labels')))
    self.num_task = self.num_task + 1
    self.task2id[task_tag] = self.num_task
    self.tasks_vocabulary[task_tag] = vocab

  def set_task(self, task_tag: str):
    #self.hidden2tag = self.classification_layers[self.task2id[task_tag]]
    self.current_task = task_tag
    self.vocab = self.tasks_vocabulary[task_tag]

  def forward(self, tokens: Dict[str, torch.Tensor], label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
    hidden2tag = self.classification_layers[self.task2id[self.current_task]]
    mask = get_text_field_mask(tokens)
    embeddings = self.word_embeddings(tokens)
    if type(self.encoder) == HashedMemoryRNN:
        output = self.encoder(embeddings, mask, mem_tokens=tokens)
    else:
        output = self.encoder(embeddings, mask)
    if type(output) == tuple:
        encoder_out, activations = output
    else:
        encoder_out = output
        activations = output
    self.activations = activations
    self.labels = label
    tag_logits = hidden2tag(encoder_out)
    output = {'logits': tag_logits, 'encoder_output': encoder_out }
    if label is not None:
      _, preds = tag_logits.max(dim=1)
      self.average(matthews_corrcoef(label.data.cpu().numpy(), preds.data.cpu().numpy()))
      self.accuracy(tag_logits, label)
      output["loss"] = self.loss_function(tag_logits, label)
      #bad.register_hooks(tag_logits)
    return output

  def get_metrics(self, reset: bool = False) -> Dict[str, float]:
    return {"accuracy": self.accuracy.get_metric(reset), "average": self.average.get_metric(reset)}

  def get_activations(self) -> []:
    return self.activations, self.labels


@Model.register("seq2seq_classifier")
class Seq2SeqClassifier(Model):
  def __init__(self, word_embeddings: TextFieldEmbedder, encoder: Seq2SeqEncoder, vocab: Vocabulary, hidden_dimension: int, bs: int) -> None:
    super().__init__(vocab)
    self.word_embeddings = word_embeddings
    self.encoder = encoder
    self.bs = bs
    self.hidden_dim = hidden_dimension
    self.vocab = vocab
    self.tasks_vocabulary = {"default": vocab}
    self.current_task = "default"
    self.num_task = 0
    self.classification_layers = torch.nn.ModuleList([torch.nn.Linear(in_features=self.hidden_dim, out_features=self.vocab.get_vocab_size('labels'))])
    self.task2id = { "default": 0 }
    self.hidden2tag = self.classification_layers[self.task2id["default"]]
    self.accuracy = CategoricalAccuracy()
    self.loss_function = torch.nn.CrossEntropyLoss()
    self.average  = Average()
    self.activations = []
    self.labels = []

  def add_task(self, task_tag: str, vocab: Vocabulary):
    self.classification_layers.append(torch.nn.Linear(in_features=self.hidden_dim, out_features=vocab.get_vocab_size('labels')))
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
    tag_logits = self.hidden2tag(torch.nn.functional.adaptive_max_pool1d(encoder_out.permute(0,2,1), (1,)).view(-1, self.hidden_dim))
    output = {'logits': tag_logits }
    self.activations = encoder_out
    self.labels = label

    if label is not None:
      _, preds = tag_logits.max(dim=1)
      self.average(matthews_corrcoef(label.data.cpu().numpy(), preds.data.cpu().numpy()))
      self.accuracy(tag_logits, label)
      output["loss"] = self.loss_function(tag_logits, label)

    return output

  def get_metrics(self, reset: bool = False) -> Dict[str, float]:
    return {"accuracy": self.accuracy.get_metric(reset), "average": self.average.get_metric(reset)}

  def get_activations(self) -> []:
    return self.activations, self.labels
