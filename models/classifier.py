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
from models.task_encoding import TaskEncoding
from models.transformer_encoder import PositionalEncoding
from models.ewc import EWC
from allennlp.nn.util import move_to_device

from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.util import move_optimizer_to_cuda

@Model.register("main_classifier")
class MainClassifier(Model):
  def __init__(self, word_embeddings: TextFieldEmbedder, encoder: Seq2VecEncoder,
               vocab: Vocabulary, args, e_dim, inv_temp: float = None,
               temp_inc:float = None, task_embed = None) -> None:
    super().__init__(vocab)
    self.args = args
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
    self.inv_temp = inv_temp
    self.temp_inc = temp_inc
    self.e_dim = e_dim
    self.task_encoder = TaskEncoding(self.e_dim) if task_embed else None
    self.pos_embedding = PositionalEncoding(self.e_dim, 0.5) if self.args.position_embed else None
    self.args = args
    self._len_dataset = None
    if self.args.ewc:
        self.ewc = EWC(self)

  def add_target_padding(self):
     self.encoder.add_target_padding()

  def add_task(self, task_tag: str, vocab: Vocabulary):
    self.classification_layers.append(torch.nn.Linear(in_features=self.encoder.get_output_dim(), out_features=vocab.get_vocab_size('labels')))
    self.num_task = self.num_task + 1
    self.task2id[task_tag] = self.num_task
    self.tasks_vocabulary[task_tag] = vocab

  def set_ewc(self, mode=True):
      if mode:
          self.loss_function = torch.nn.NLLLoss()
      else:
          self.loss_function = torch.nn.CrossEntropyLoss()

  def set_task(self, task_tag: str, training: bool = False, normaliser = None):
    #self.hidden2tag = self.classification_layers[self.task2id[task_tag]]
    self.training = training
    self.current_task = task_tag
    if training and (not normaliser is None):
        self._len_dataset = normaliser
    self.vocab = self.tasks_vocabulary[task_tag]
    if training and self.temp_inc:
        self.inv_temp = self.temp_inc*self.inv_temp

  def get_current_taskid(self):
      return self.task2id[self.current_task]

  def forward(self, tokens: Dict[str, torch.Tensor], label: torch.Tensor = None, task_id = None) -> Dict[str, torch.Tensor]:
    if (task_id is None):
        task_id = self.get_current_taskid()
    hidden2tag = self.classification_layers[task_id]
    if torch.cuda.is_available():
        tokens = move_to_device(tokens, torch.cuda.current_device())
    mask = get_text_field_mask(tokens)
    embeddings = self.word_embeddings(tokens)
    if self.args.position_embed:
        embeddings = self.pos_embedding(embeddings)
    if self.task_encoder:
        bs,seq,edi = embeddings.shape
        task_em = torch.randn((bs,seq,1))
        task_em.fill_(self.get_current_taskid())
        if torch.cuda.is_available():
            task_em = move_to_device(task_em, torch.cuda.current_device())
        embeddings = torch.cat([embeddings, task_em], dim=-1)
        #embeddings = self.task_encoder(embeddings, self.get_current_taskid())

    if self.inv_temp:
        embeddings = self.inv_temp*embeddings
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
      if self.args.ewc and self.training:
          output["loss"] += self.args.ewc_importance*self.ewc.penalty(self.get_current_taskid())
          output["loss"].backward(retain_graph=True)
          self.ewc.update_penalty(self.task2id[self.current_task], self, self._len_dataset)
    return output

  def get_metrics(self, reset: bool = False) -> Dict[str, float]:
    return {"accuracy": self.accuracy.get_metric(reset), "average": self.average.get_metric(reset)}

  def get_activations(self) -> []:
    return self.activations, self.labels
