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
from models.mean_classifier import MeanClassifier
from models.trec import TrecDatasetReader
from models.subjectivity import SubjectivityDatasetReader
from models.CoLA import CoLADatasetReader
from models.ag import AGNewsDatasetReader
import models.utils as utils
import argparse
#from torch.utils.tensorboard import SummaryWriter

majority = {'subjectivity': 0.5, 'sst': 0.2534059946, 'trec': 0.188, 'cola': 0.692599620493358, 'ag': 0.25, 'sst_2c': 0.51}

sota = {'subjectivity': 0.955, 'sst': 0.547, 'trec': 0.9807, 'cola': 0.772, 'ag' : 0.955 , 'sst_2c': 0.968}


parser = argparse.ArgumentParser(description='Argument for catastrophic training.')
parser.add_argument('--task', action='append', help="Task to be added to model, put each task seperately, Allowed tasks currently are : \nsst \ncola \ntrec \nsubjectivity. If train and evaluate options are not provide they default to tasks option.\n")
parser.add_argument('--train', action='append', help="Task to train on, put each task seperately, Allowed tasks currently are : \nsst \ncola \ntrec \nsubjectivity\n")
parser.add_argument('--evaluate', action='append', help="Task to evaluate on, put each task seperately, Allowed tasks currently are : \nsst \ncola \ntrec \nsubjectivity\n")
parser.add_argument('--few_shot', action='store_true', help="Train task on few shot learning before evaluating.")
parser.add_argument('--mean_classifier', action='store_true', help="Start using mean classifier instead of normal evaluation.")
parser.add_argument('--joint', action='store_true', help="Do the joint training or by the task sequentially")
parser.add_argument('--diff_class', action='store_true', help="Do training with Different classifier for each task")
parser.add_argument('--cnn', action='store_true', help="Use CNN")
parser.add_argument('--pyramid', action='store_true', help="Use Deep Pyramid CNN works only when --cnn is applied")
parser.add_argument('--epochs', type=int, default=1000, help="Number of epochs to train for")
parser.add_argument('--layers', type=int, default=1, help="Number of layers")
parser.add_argument('--dropout', type=float, default=0, help="Use dropout")
parser.add_argument('--bs', type=float, default=128, help="Batch size to use")

# Optimization Based Parameters
parser.add_argument('--wdecay', type=float, help="L2 Norm to use")
parser.add_argument('--lr', type=float, default=0.001, help="L2 Norm to use")
parser.add_argument('--opt_alg', type=str, default="adam", help="Optimization algorithm to use")
parser.add_argument('--patience', type=int, default=10, help="Number of layers")


parser.add_argument('--e_dim', type=int, default=128, help="Embedding Dimension")
parser.add_argument('--h_dim', type=int, default=1150, help="Hidden Dimension")
parser.add_argument('--s_dir', help="Serialization directory")
parser.add_argument('--seq2vec', help="Use Sequence to sequence",action='store_true')
parser.add_argument('--gru', help="Use GRU UNIt",action='store_true')
parser.add_argument('--majority', help="Use Sequence to sequence",action='store_true')
parser.add_argument('--tryno', type=int, default=1, help="This is ith try add this to name of df")
parser.add_argument('--run_name', type=str, default="Default", help="This is the run name being saved to tensorboard")
parser.add_argument('--storage_prefix', type=str, default="./runs/", help="This is used to store the runs inside runs folder")



parser.add_argument('--pooling', type=str, default="max", help="Selects the pooling operation for CNN, max pooling, min pooling, average pooling. max,min,avg")
parser.add_argument('--no_save_weight', action='store_true', help="Disable saving of weights")

args = parser.parse_args()


#writer=SummaryWriter(run_name)
#writer.add_text("args", str(args))


print("Training on these tasks", args.task, 
      "\nJoint", args.joint,
      "\nepochs", args.epochs,
      "\nlayers", args.layers,
      "\ndropout", args.dropout,
      "\ne_dim", args.e_dim,
      "\nh_dim", args.h_dim,
      "\ndiff_class", args.diff_class)
tasks = list(args.task)

train_data = {}
dev_data = {}
few_data = {}
vocabulary = {}

for task in tasks:
  utils.load_dataset(task, train_data, dev_data, few_data)

if not args.train:
   print("Train option not provided,  defaulting to tasks")
   train = tasks
else:
   train = list(args.train)

if not args.evaluate:
   print("evaluate option not provided,  defaulting to tasks")
   evaluate_tasks = tasks
else:
   evaluate_tasks = list(args.evaluate)

print(type(train_data[tasks[0]]))
joint_train = []
joint_dev = []
task_code=""
labels_mapping={}
for i in tasks:
  task_code+=str("_"+str(i))
  joint_train += train_data[i]
  joint_dev += dev_data[i]
  if args.diff_class:
    vocabulary[i] = Vocabulary.from_instances(train_data[i] + dev_data[i])
    label_size = vocabulary[i].get_vocab_size('labels')
    label_indexes = {}
    for si in range(label_size):
      label_indexes[si] = vocabulary[i].get_token_from_index(si, 'labels')
    labels_mapping[i] = label_indexes
print("Labels mapping :  ", labels_mapping)


train_code=""
for i in train:
  train_code += str("_"+str(i))


evaluate_code=""
for i in evaluate_tasks:
  evaluate_code += str("_"+str(i))


if args.few_shot:
  task_code += ('_train_' + train_code)
  task_code += ('_evaluate_' + evaluate_code)

## Define Run Name and args to tensorboard for tracking.
run_name=args.storage_prefix + args.run_name+"_"+str(args.layers)+"_hdim_"+str(args.h_dim)+"_code_"+task_code+"/run_"+str(args.tryno)

vocab = Vocabulary.from_instances(joint_train + joint_dev)

#vocab.print_statistics()


token_embeddings = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
	  embedding_dim=args.e_dim)

word_embeddings = BasicTextFieldEmbedder({"tokens": token_embeddings})


experiment="lstm"
print("CNN",args.cnn)
if args.cnn:
  experiment="cnn_"
  experiment += args.pooling
  print(" Going CNN",args.cnn)
  ngrams_f=(2,)
  cnn = CnnEncoder(embedding_dim=args.e_dim,
                   num_layers=args.layers,
		   ngram_filter_sizes=ngrams_f,
		   num_filters=args.h_dim,
                   pooling=args.pooling)
  if args.pyramid:
      experiment="dpcnn"
      cnn = DeepPyramidCNN(embedding_dim=args.e_dim,
                       num_layers=args.layers,
		       ngram_filter_sizes=ngrams_f,
		       num_filters=args.h_dim)
  model = MainClassifier(word_embeddings, cnn, vocab)
  if args.mean_classifier:
    print("We are on journey to use the mean classifier now.")
    model = MeanClassifier(word_embeddings, cnn, vocab)
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

if not args.no_save_weight:
    save_weight = SaveWeights(experiment, args.layers, args.h_dim, task_code, labels_mapping, args.mean_classifier)

for i in tasks:
  model.add_task(i, vocabulary[i])
if torch.cuda.is_available():
  model.cuda(0)

optimizer = utils.get_optimizer(args.opt_alg, model.parameters(), args.lr, args.wdecay)

torch.set_num_threads(4)
iterator = BucketIterator(batch_size=args.bs, sorting_keys=[("tokens", "num_tokens")])

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
  for i in evaluate_tasks:
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
                  num_serialized_models_to_keep=1,
                  model_save_interval=1,
                  serialization_dir=run_name,
                  histogram_interval=100,
                  patience=args.patience,
                  num_epochs=args.epochs,
                  cuda_device=devicea)
  for tid,i in enumerate(train,1):
    print("\nTraining task ", i)
    sys.stdout.flush()
    if args.diff_class:
      model.set_task(i)
      trainer._num_epochs = args.epochs
      iterator.index_with(vocabulary[i])
      trainer.train_data = train_data[i]
      trainer._validation_data = dev_data[i]
      trainer.model = model
      trainer.iterator = iterator
      trainer._validation_iterator = iterator
      trainer._metric_tracker.clear()
    if not args.majority:
      metrics = trainer.train()
      trainer._tensorboard.add_train_scalar("restore_checkpoint/"+str(i), metrics['training_epochs'], timestep=tid)
    for j in evaluate_tasks:
      print("\nEvaluating ", j)
      sys.stdout.flush()
      if args.diff_class:
        model.set_task(j)
        # This batch size of 10000 is hack to get activation while doing evaluation.
        iterator1 = BucketIterator(batch_size=10000, sorting_keys=[("tokens", "num_tokens")])
        iterator1.index_with(vocabulary[j])
        if args.few_shot:
          met = evaluate(model=model,
	                    instances=dev_data[j],
	                    data_iterator=iterator1,
	                    cuda_device=devicea,
	                    batch_weight_key=None)
          print("Now few_shot training ", j, "Metric before ", met," \n")
          for name, param in model.named_parameters():
            print("Named parameters for freezing ", name)
            if name.startswith('encoder') or name.startswith('word_embeddings'):
              print("Freezing param ", name)
              param.requires_grad = False
          iterator1 = BucketIterator(batch_size=args.bs, sorting_keys=[("tokens", "num_tokens")])
          iterator1.index_with(vocabulary[j])
          trainer.model = model
          trainer.train_data = few_data[j]
          trainer._validation_data = few_data[j]
          trainer.iterator = iterator1
          trainer._metric_tracker.clear()
          print("Doing few shot traing current things is ", trainer._metric_tracker._epochs_with_no_improvement, trainer._metric_tracker._epoch_number)
          trainer._num_epochs = 10
          trainer.train()
          # Back to hack of 10000 to get all the activations together as
          iterator1 = BucketIterator(batch_size=10000, sorting_keys=[("tokens", "num_tokens")])
          iterator1.index_with(vocabulary[j])
          trainer._num_epochs = args.epochs
      if args.mean_classifier:
        model.adding_mean_representation = True
        metric = evaluate(model=model,
	                   instances=few_data[j],
	                   data_iterator=iterator1,
	                   cuda_device=devicea,
	                   batch_weight_key=None)
        model.adding_mean_representation = False
        model.get_mean_prune_sampler()
        model.evaluate_using_mean = True
      print("Now evaluating ", j)
      metric = evaluate(model=model,
	 instances=dev_data[j],
	 data_iterator=iterator1,
	 cuda_device=devicea,
	 batch_weight_key=None)

      # Take first 500 instances for evaluating activations.
      evaluate(model=model,
	 instances=dev_data[j][:500],
	 data_iterator=iterator1,
	 cuda_device=devicea,
	 batch_weight_key=None)

      if not args.no_save_weight:
          save_weight.add_activations(model,i,j)

      if args.mean_classifier:
        model.evaluate_using_mean = False

      if j == 'cola':
          metric['metric'] = metric['average']
      else:
          metric['metric'] = metric['accuracy']

      standard_metric = (float(metric['metric']) - majority[j]) / (sota[j] - majority[j])
      if j not in overall_metrics:
        overall_metrics[j] = {}
        overall_metrics[j][i] = metric
        ostandard_metrics[j] = {}
        ostandard_metrics[j][i] = standard_metric
      else:
        overall_metrics[j][i] = metric
        ostandard_metrics[j][i] = standard_metric
      print("Adding timestep to trainer",tid, tasks, j, float(metric['metric']))
      trainer._tensorboard.add_train_scalar("evaluate/"+str(j), float(metric['metric']), timestep=tid)
      trainer._tensorboard.add_train_scalar("standard_evaluate/"+str(j), standard_metric, timestep=tid)
    if not args.majority:
      print("\n Joint Evaluating ")
      sys.stdout.flush()
      model.set_task("default")
  #    overall_metric = evaluate(model=model,
  #       instances=joint_dev,
  #       data_iterator=iterator,
  #       cuda_device=devicea,
  #       batch_weight_key=None)
  #    overall_metrics[i]["Joint"] = overall_metric
  # Calculate the catastrophic forgetting and add it into tensorboard before
  # closing the tensorboard
  c_standard_metric = get_catastrophic_metric(train, ostandard_metrics)
  print("Forgetting Results", c_standard_metric)
  for tid,task in enumerate(c_standard_metric, 1):
      trainer._tensorboard.add_train_scalar("forgetting_metric/standard_" + task,
					    c_standard_metric[task],
					    timestep=tid)
  if not args.no_save_weight:
      save_weight.write_activations(overall_metrics, trainer, tasks)

  trainer._tensorboard._train_log.close()

if not args.diff_class:
  print("\n Joint Evaluating ")
  sys.stdout.flush()
  overall_metric = evaluate(model=model,
	 instances=joint_dev,
	 data_iterator=iterator,
	 cuda_device=devicea,
	 batch_weight_key=None)

print("Training Results are on these :", args.task,
      "\nJoint", args.joint,
      "\nepochs", args.epochs,
      "\nlayers", args.layers,
      "\ndropout", args.dropout,
      "\ne_dim", args.e_dim,
      "\nh_dim", args.h_dim,
      "\ndiff_class", args.diff_class)

print("Accuracy and Loss")
header="Accuracy"
for i in evaluate_tasks:
  header = header + "\t\t" + i
insert_in_pandas_list=[]
print(header)
for d in train:
  print_data=d
  insert_pandas_dict={'code': task_code, 'layer': args.layers, 'h_dim': args.h_dim, 'task': d, 'try': args.tryno, 'experiment': experiment, 'metric': 'accuracy'}
  i=0
  for k in evaluate_tasks:
    print_data = print_data + "\t\t" + str(overall_metrics[k][d]["metric"])
    insert_pandas_dict[k] = overall_metrics[k][d]["metric"]
  insert_in_pandas_list.append(insert_pandas_dict)
  print(print_data)
print("\n\n")
initial_path="dfs/Results" + args.run_name
if not args.seq2vec:
  intial_path="dfs_att/Results_selfattention" + args.run_name
if args.cnn:
  initial_path="dfs/Results_CNN_" + args.run_name
if args.gru:
  initial_path="dfs/Results_GRU_" + args.run_name

df=pd.DataFrame(insert_in_pandas_list)

if args.few_shot:
  initial_path += ('_train_' + train_code)
  initial_path += ('_evaluate_' + evaluate_code)

df.to_pickle(path=str(initial_path+task_code+"_"+str(args.layers)+"_"+str(args.h_dim)+"_"+str(args.tryno)+".df"))
#print(joint_print_data)
