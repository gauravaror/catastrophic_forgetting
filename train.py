from typing import Iterator, List, Dict
import torch
import torch.optim as optim
import numpy as np
import sys

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
from allennlp.training.util import move_optimizer_to_cuda, evaluate
from allennlp.common.params import Params
from models.classifier import MainClassifier
from models.trec import TrecDatasetReader
from models.CoLA import CoLADatasetReader
import argparse


parser = argparse.ArgumentParser(description='Argument for catastrophic training.')
parser.add_argument('--task', action='append')
parser.add_argument('--joint', action='store_true')
parser.add_argument('--epochs', type=int, default=1000)
args = parser.parse_args()

print("Training on these tasks", args.task, 
      "\nJoint", args.joint,
      "\nepochs", args.epochs)



reader_senti = StanfordSentimentTreeBankDatasetReader()
reader_cola = CoLADatasetReader()
reader_trec = TrecDatasetReader()

train_data = {}
dev_data = {}
train_data["sst"] = reader_senti.read('data/SST/trees/train.txt')
dev_data["sst"] = reader_senti.read('data/SST/trees/dev.txt')

train_data["cola"] = reader_cola.read('data/CoLA/train.txt')
dev_data["cola"] = reader_cola.read('data/CoLA/dev.txt')

train_data["trec"] = reader_trec.read('data/TREC/train.txt')
dev_data["trec"] = reader_trec.read('data/TREC/dev.txt')

train_data["subjectivity"] = reader_trec.read('data/Subjectivity/train.txt')
dev_data["subjectivity"] = reader_trec.read('data/Subjectivity/test.txt')

tasks = args.task

print(type(train_data[tasks[0]]))
joint_train = []
joint_dev = []
for i in tasks:
  joint_train += train_data[i]
  joint_dev += dev_data[i]
vocab = Vocabulary.from_instances(joint_train + joint_dev)

vocab.print_statistics()

EMBEDDING_DIM = 128
HIDDEN_DIM = 1150


token_embeddings = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
	  embedding_dim=EMBEDDING_DIM)

word_embeddings = BasicTextFieldEmbedder({"tokens": token_embeddings})

lstm = PytorchSeq2VecWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))

model = MainClassifier(word_embeddings, lstm, vocab)

model.cuda(0)

optimizer = optim.SGD(model.parameters(), lr=0.1)

move_optimizer_to_cuda(optimizer)

torch.set_num_threads(8)
iterator = BucketIterator(batch_size=32, sorting_keys=[("tokens", "num_tokens")])

iterator.index_with(vocab)

if args.joint:
  print("\nTraining task : Joint ", tasks)
  trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=joint_train,
                  validation_dataset=joint_dev,
                  patience=10,
                  num_epochs=args.epochs,
		  cuda_device=0)
  trainer.train()
  for i in tasks:
    print("\nEvaluating ", i)
    sys.stdout.flush()
    evaluate(model=model,
	 instances=dev_data[i],
	 data_iterator=iterator,
	 cuda_device=0,
	 batch_weight_key=None)
else:
  for i in tasks:
    print("\nTraining task ", i)
    trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_data[i],
                  validation_dataset=dev_data[i],
                  patience=10,
                  num_epochs=args.epochs,
		  cuda_device=0)
    trainer.train()
    for j in tasks:
      print("\nEvaluating ", j)
      sys.stdout.flush()
      evaluate(model=model,
	 instances=dev_data[j],
	 data_iterator=iterator,
	 cuda_device=0,
	 batch_weight_key=None)

print("\n Joint Evaluating ")
sys.stdout.flush()
evaluate(model=model,
	 instances=joint_dev,
	 data_iterator=iterator,
	 cuda_device=0,
	 batch_weight_key=None)
