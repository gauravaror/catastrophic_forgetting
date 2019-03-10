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
parser.add_argument('--task', action='append', help="Task to train on, put each task seperately, Allowed tasks currently are : \nsst \ncola \ntrec \nsubjectivity\n")
parser.add_argument('--joint', action='store_true', help="Do the joint training or by the task sequentially")
parser.add_argument('--diff_class', action='store_true', help="Do training with Different classifier for each task")
parser.add_argument('--epochs', type=int, default=1000, help="Number of epochs to train for")
parser.add_argument('--layers', type=int, default=1, help="Number of layers")
parser.add_argument('--dropout', type=float, default=0, help="Use dropout")
parser.add_argument('--e_dim', type=int, default=128, help="Embedding Dimension")
parser.add_argument('--h_dim', type=int, default=1150, help="Hidden Dimension")

args = parser.parse_args()

print("Training on these tasks", args.task, 
      "\nJoint", args.joint,
      "\nepochs", args.epochs,
      "\nlayers", args.layers,
      "\dropout", args.dropout,
      "\ne_dim", args.e_dim,
      "\nh_dim", args.h_dim,
      "\ndiff_class", args.diff_class)


reader_senti = StanfordSentimentTreeBankDatasetReader()
reader_cola = CoLADatasetReader()
reader_trec = TrecDatasetReader()

train_data = {}
dev_data = {}
vocabulary = {}

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
  if args.diff_class:
    vocabulary[i] = Vocabulary.from_instances(train_data[i] + dev_data[i])

vocab = Vocabulary.from_instances(joint_train + joint_dev)

vocab.print_statistics()


token_embeddings = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
	  embedding_dim=args.e_dim)

word_embeddings = BasicTextFieldEmbedder({"tokens": token_embeddings})

lstm = PytorchSeq2VecWrapper(torch.nn.LSTM(args.e_dim, args.h_dim,
					   num_layers=args.layers,
					   dropout=args.dropout,
					   batch_first=True))

model = MainClassifier(word_embeddings, lstm, vocab)

for i in tasks:
  model.add_task(i, vocabulary[i])
model.cuda(0)

optimizer = optim.SGD(model.parameters(), lr=0.1)

move_optimizer_to_cuda(optimizer)

torch.set_num_threads(8)
iterator = BucketIterator(batch_size=32, sorting_keys=[("tokens", "num_tokens")])

iterator.index_with(vocab)

overall_metrics = {}
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
    sys.stdout.flush()
    if args.diff_class:
      model.set_task(i)
      iterator.index_with(vocabulary[i])
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
      if args.diff_class:
        model.set_task(j)
        iterator.index_with(vocabulary[j])
      metric = evaluate(model=model,
	 instances=dev_data[j],
	 data_iterator=iterator,
	 cuda_device=0,
	 batch_weight_key=None)
      overall_metrics["train_"+i+"_evaluate_"+j] = metric

if not args.diff_class:
  print("\n Joint Evaluating ")
  sys.stdout.flush()
  evaluate(model=model,
	 instances=joint_dev,
	 data_iterator=iterator,
	 cuda_device=0,
	 batch_weight_key=None)

print("Training on these tasks", args.task, 
      "\nJoint", args.joint,
      "\nepochs", args.epochs,
      "\nlayers", args.layers,
      "\dropout", args.dropout,
      "\ne_dim", args.e_dim,
      "\nh_dim", args.h_dim,
      "\ndiff_class", args.diff_class)

for d in overall_metrics.keys():
  current_metrics = overall_metrics[d]
  print("Evaluation for : ", d)
  for key, metric in current_metrics.items():
    print(key," : " ,metric)

