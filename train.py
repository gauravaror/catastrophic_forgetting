from typing import Iterator, List, Dict
import torch
import torch.optim as optim
import numpy as np
import sys
import pandas as pd
import random

from models.utils import get_catastrophic_metric
from allennlp.data.dataset_readers import DatasetReader
from models.sst import StanfordSentimentTreeBankDatasetReader1
from allennlp.data.vocabulary import Vocabulary
from models.save_weights import SaveWeights

from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.models import Model

from allennlp.modules.seq2seq_encoders.stacked_self_attention import StackedSelfAttentionEncoder
from models.cnn_encoder import CnnEncoder

from models.deep_pyramid_cnn import DeepPyramidCNN
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.modules.seq2seq_encoders.multi_head_self_attention import MultiHeadSelfAttention
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.util import move_optimizer_to_cuda, evaluate
from allennlp.common.params import Params
from models.classifier import MainClassifier, Seq2SeqClassifier, MajorityClassifier
from models.trec import TrecDatasetReader
from models.CoLA import CoLADatasetReader
import argparse
#from torch.utils.tensorboard import SummaryWriter

majority = {'subjectivity': 0.5, 'sst': 0.2534059946, 'trec': 0.188, 'cola': 0.692599620493358}

sota = {'subjectivity': 0.955, 'sst': 0.547, 'trec': 0.9807, 'cola': 0.772}


parser = argparse.ArgumentParser(description='Argument for catastrophic training.')
parser.add_argument('--task', action='append', help="Task to train on, put each task seperately, Allowed tasks currently are : \nsst \ncola \ntrec \nsubjectivity\n")
parser.add_argument('--joint', action='store_true', help="Do the joint training or by the task sequentially")
parser.add_argument('--diff_class', action='store_true', help="Do training with Different classifier for each task")
parser.add_argument('--cnn', action='store_true', help="Use CNN")
parser.add_argument('--pyramid', action='store_true', help="Use Deep Pyramid CNN works only when --cnn is applied")
parser.add_argument('--epochs', type=int, default=1000, help="Number of epochs to train for")
parser.add_argument('--layers', type=int, default=1, help="Number of layers")
parser.add_argument('--patience', type=int, default=10, help="Number of layers")
parser.add_argument('--dropout', type=float, default=0, help="Use dropout")
parser.add_argument('--e_dim', type=int, default=128, help="Embedding Dimension")
parser.add_argument('--h_dim', type=int, default=1150, help="Hidden Dimension")
parser.add_argument('--s_dir', help="Serialization directory")
parser.add_argument('--seq2vec', help="Use Sequence to sequence",action='store_true')
parser.add_argument('--gru', help="Use GRU UNIt",action='store_true')
parser.add_argument('--majority', help="Use Sequence to sequence",action='store_true')
parser.add_argument('--tryno', type=int, default=1, help="This is ith try add this to name of df")
parser.add_argument('--run_name', type=str, default="Default", help="This is the run name being saved to tensorboard")

args = parser.parse_args()


#writer=SummaryWriter(run_name)
#writer.add_text("args", str(args))


print("Training on these tasks", args.task, 
      "\nJoint", args.joint,
      "\nepochs", args.epochs,
      "\nlayers", args.layers,
      "\dropout", args.dropout,
      "\ne_dim", args.e_dim,
      "\nh_dim", args.h_dim,
      "\ndiff_class", args.diff_class)


reader_senti = StanfordSentimentTreeBankDatasetReader1()
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

tasks = list(args.task)

print(type(train_data[tasks[0]]))
joint_train = []
joint_dev = []
task_code=""
for i in tasks:
  task_code+=str("_"+str(i))
  joint_train += train_data[i]
  joint_dev += dev_data[i]
  if args.diff_class:
    vocabulary[i] = Vocabulary.from_instances(train_data[i] + dev_data[i])
## Define Run Name and args to tensorboard for tracking.
run_name="runs/"+args.run_name+"_"+str(args.layers)+"_hdim_"+str(args.h_dim)+"_code_"+task_code+"/run_"+str(args.tryno)

vocab = Vocabulary.from_instances(joint_train + joint_dev)

vocab.print_statistics()


token_embeddings = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
	  embedding_dim=args.e_dim)

word_embeddings = BasicTextFieldEmbedder({"tokens": token_embeddings})


experiment="lstm"
print("CNN",args.cnn)
if args.cnn:
  experiment="cnn"
  print(" Going CNN",args.cnn)
  ngrams_f=(2,)
  cnn = CnnEncoder(embedding_dim=args.e_dim,
                   num_layers=args.layers,
		   ngram_filter_sizes=ngrams_f,
		   num_filters=args.h_dim)
  if args.pyramid:
      experiment="dpcnn"
      cnn = DeepPyramidCNN(embedding_dim=args.e_dim,
                       num_layers=args.layers,
		       ngram_filter_sizes=ngrams_f,
		       num_filters=args.h_dim)
  model = MainClassifier(word_embeddings, cnn, vocab)
elif args.seq2vec or args.majority:
  experiment="lstm"
  lstm = PytorchSeq2VecWrapper(torch.nn.LSTM(args.e_dim, args.h_dim,
					   num_layers=args.layers,
					   dropout=args.dropout,
					   batch_first=True))
  print(" Going LSTM",args.cnn)
  lstmseq = PytorchSeq2SeqWrapper(torch.nn.LSTM(args.e_dim, args.h_dim,
					   num_layers=args.layers,
					   dropout=args.dropout,
					   batch_first=True))
  if args.gru:
    experiment="gru"
    lstm = PytorchSeq2VecWrapper(torch.nn.GRU(args.e_dim, args.h_dim,
					   num_layers=args.layers,
					   dropout=args.dropout,
					   batch_first=True))
  model = MainClassifier(word_embeddings, lstm, vocab)
  if args.majority:
    model = MajorityClassifier(vocab)
else:
  experiment="selfattention"
  print(" Going Attention",args.cnn)
  attentionseq = StackedSelfAttentionEncoder(
					   input_dim=args.e_dim,
					   hidden_dim=args.h_dim,
					   projection_dim=128,
					   feedforward_hidden_dim=128,
					   num_layers=args.layers,
					   num_attention_heads=8,
					   attention_dropout_prob=args.dropout)
  model = Seq2SeqClassifier(word_embeddings, attentionseq, vocab, hidden_dimension=args.h_dim, bs=32)

save_weight = SaveWeights(experiment, args.layers, args.h_dim, task_code)

for i in tasks:
  model.add_task(i, vocabulary[i])
if torch.cuda.is_available():
  model.cuda(0)

optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)

if torch.cuda.is_available():
  move_optimizer_to_cuda(optimizer)

torch.set_num_threads(4)
iterator = BucketIterator(batch_size=32, sorting_keys=[("tokens", "num_tokens")])

iterator.index_with(vocab)
devicea = -1
if torch.cuda.is_available():
  devicea = 0

overall_metrics = {}
ostandard_metrics = {}
if args.joint:
  print("\nTraining task : Joint ", tasks)
  trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=joint_train,
                  validation_dataset=joint_dev,
                  patience=args.patience,
                  num_epochs=args.epochs,
		  cuda_device=devicea)
  trainer.train()
  for i in tasks:
    print("\nEvaluating ", i)
    sys.stdout.flush()
    evaluate(model=model,
	 instances=dev_data[i],
	 data_iterator=iterator,
	 cuda_device=devicea,
	 batch_weight_key=None)
else:
  trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_data[i],
                  validation_dataset=dev_data[i],
		  num_serialized_models_to_keep=0,
		  serialization_dir=run_name,
		  histogram_interval=100,
                  patience=args.patience,
                  num_epochs=args.epochs,
		  cuda_device=devicea)
  for tid,i in enumerate(tasks,1):
    print("\nTraining task ", i)
    sys.stdout.flush()
    if args.diff_class:
      model.set_task(i)
      iterator.index_with(vocabulary[i])
      trainer.train_data = train_data[i]
      trainer._validation_data = dev_data[i]
      trainer.model = model
      trainer.iterator = iterator
      trainer._validation_iterator = iterator
      trainer._metric_tracker.clear()
      trainer._tensorboard.add_train_scalar("restore_checkpoint/"+str(i), 1)
    if not args.majority:
      trainer.train()
    #save_weight.write_weights_new(model, args.layers, args.h_dim, task_code, i, args.tryno)
    for j in tasks:
      print("\nEvaluating ", j)
      sys.stdout.flush()
      if args.diff_class:
        model.set_task(j)
        iterator1 = BucketIterator(batch_size=10000, sorting_keys=[("tokens", "num_tokens")])
        iterator1.index_with(vocabulary[j])
      metric = evaluate(model=model,
	 instances=dev_data[j],
	 data_iterator=iterator1,
	 cuda_device=devicea,
	 batch_weight_key=None)
      save_weight.add_activations(model,i,j)
      standard_metric = (float(metric['accuracy']) - majority[j]) / (sota[j] - majority[j])
      if i not in overall_metrics:
        overall_metrics[i] = {}
        overall_metrics[i][j] = metric
        ostandard_metrics[i] = {}
        ostandard_metrics[i][j] = standard_metric
      else:
        overall_metrics[i][j] = metric
        ostandard_metrics[i][j] = standard_metric
      print("Adding timestep to trainer",tid, tasks, j, float(metric['accuracy']))
      trainer._tensorboard.add_train_scalar("evaluate/"+str(j), float(metric['accuracy']), timestep=tid)
      trainer._tensorboard.add_train_scalar("standard_evaluate/"+str(j), standard_metric, timestep=tid)
    if not args.majority:
      print("\n Joint Evaluating ")
      sys.stdout.flush()
      model.set_task("default")
      overall_metric = evaluate(model=model,
         instances=joint_dev,
         data_iterator=iterator,
         cuda_device=devicea,
         batch_weight_key=None)
      overall_metrics[i]["Joint"] = overall_metric

  # Calculate the catastrophic forgetting and add it into tensorboard before
  # closing the tensorboard
  c_standard_metric = get_catastrophic_metric(tasks, ostandard_metrics)
  for tid,task in enumerate(c_standard_metric, 1):
      trainer._tensorboard.add_train_scalar("forgetting_metric/standard_"+ task,
					    c_standard_metric[task],
					    timestep=tid)
  trainer._tensorboard._train_log.close()

if not args.diff_class:
  print("\n Joint Evaluating ")
  sys.stdout.flush()
  overall_metric = evaluate(model=model,
	 instances=joint_dev,
	 data_iterator=iterator,
	 cuda_device=devicea,
	 batch_weight_key=None)

save_weight.write_activations(overall_metrics)
print("Training on these tasks", args.task, 
      "\nJoint", args.joint,
      "\nepochs", args.epochs,
      "\nlayers", args.layers,
      "\ndropout", args.dropout,
      "\ne_dim", args.e_dim,
      "\nh_dim", args.h_dim,
      "\ndiff_class", args.diff_class)

print("Accuracy and Loss")
header="Accuracy"
for i in tasks:
  header = header + "\t" + i
insert_in_pandas_list=[]
print(header)
for d in tasks:
  current_metrics = overall_metrics[d]
  print_data=d
  insert_pandas_dict={'code': task_code, 'layer': args.layers, 'h_dim': args.h_dim, 'task': d, 'try': args.tryno, 'experiment': experiment, 'metric': 'accuracy'}
  i=0
  for k in tasks:
    print_data = print_data + "\t" + str(overall_metrics[k][d]["accuracy"])
    insert_pandas_dict[k] = overall_metrics[k][d]["accuracy"]
  insert_in_pandas_list.append(insert_pandas_dict)
  print(print_data)
joint_print_data = "Joint\t"
#for o in tasks:
#  joint_print_data = joint_print_data + "\t" + str(overall_metrics[o]["Joint"]["accuracy"])
print(joint_print_data)
print("\n\n")
initial_path="dfs/Results"
if not args.seq2vec:
  intial_path="dfs_att/Results_selfattention"
if args.cnn:
  initial_path="dfs/Results_CNN_"
if args.gru:
  initial_path="dfs/Results_GRU_"

header="Loss"
for i in tasks:
  header = header + "\t" + i
print(header)
for d in tasks:
  current_metrics = overall_metrics[d]
  insert_pandas_dict={'code': task_code, 'layer': args.layers, 'h_dim': args.h_dim, 'task': d, 'try': args.tryno, 'experiment': experiment, 'metric': 'average'}
  print_data=d
  for k in tasks:
    print_data = print_data + "\t" + str(overall_metrics[k][d]["average"])
    insert_pandas_dict[k] = overall_metrics[k][d]["average"]
  insert_in_pandas_list.append(insert_pandas_dict)
  print(print_data)
joint_print_data = "Joint\t"
for o in tasks:
  joint_print_data = joint_print_data + "\t" + str(overall_metrics[o]["Joint"]["loss"])
df=pd.DataFrame(insert_in_pandas_list)
df.to_pickle(path=str(initial_path+task_code+"_"+str(args.layers)+"_"+str(args.h_dim)+"_"+str(args.tryno)+".df"))
print(joint_print_data)
