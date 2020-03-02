from typing import Iterator, List, Dict
import torch
import torch.optim as optim
import numpy as np
import sys
import pandas as pd
import random

from models.utils import get_catastrophic_metric
from allennlp.data.vocabulary import Vocabulary
from models.save_weights import SaveWeights

from allennlp.training.metrics import CategoricalAccuracy
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.util import move_optimizer_to_cuda, evaluate
from allennlp.common.params import Params

import models.net as net
import models.utils as utils
import argparse
#from torch.utils.tensorboard import SummaryWriter

majority = {'subjectivity': 0.5, 'sst': 0.2534059946, 'trec': 0.188, 'cola': 0, 'ag': 0.25, 'sst_2c': 0.51}

sota = {'subjectivity': 0.955, 'sst': 0.547, 'trec': 0.9807, 'cola': 0.341, 'ag' : 0.955 , 'sst_2c': 0.968}


parser = argparse.ArgumentParser(description='Argument for catastrophic training.')
parser.add_argument('--task', action='append', help="Task to be added to model, put each task seperately, Allowed tasks currently are : \nsst \ncola \ntrec \nsubjectivity. If train and evaluate options are not provide they default to tasks option.\n")
parser.add_argument('--train', action='append', help="Task to train on, put each task seperately, Allowed tasks currently are : \nsst \ncola \ntrec \nsubjectivity\n")
parser.add_argument('--evaluate', action='append', help="Task to evaluate on, put each task seperately, Allowed tasks currently are : \nsst \ncola \ntrec \nsubjectivity\n")
parser.add_argument('--few_shot', action='store_true', help="Train task on few shot learning before evaluating.")
parser.add_argument('--mean_classifier', action='store_true', help="Start using mean classifier instead of normal evaluation.")
parser.add_argument('--joint', action='store_true', help="Do the joint training or by the task sequentially")
parser.add_argument('--diff_class', action='store_true', help="Do training with Different classifier for each task")

# CNN Params
parser.add_argument('--cnn', action='store_true', help="Use CNN")
parser.add_argument('--lstm', action='store_true', help="Use LSTM architecture")
parser.add_argument('--pyramid', action='store_true', help="Use Deep Pyramid CNN works only when --cnn is applied")
parser.add_argument('--ngram_filter', type=int, default=2, help="Ngram filter size to send in")
parser.add_argument('--stride', type=int, default=1, help="Strides to use for CNN")


parser.add_argument('--epochs', type=int, default=1000, help="Number of epochs to train for")
parser.add_argument('--layers', type=int, default=1, help="Number of layers")
parser.add_argument('--dropout', type=float, default=0, help="Use dropout")
parser.add_argument('--bs', type=int, default=64, help="Batch size to use")
parser.add_argument('--bidirectional', action='store_true', help="Run LSTM Network using bi-directional network.")
parser.add_argument('--embeddings', help="Use which embedding ElMO embeddings or BERT",type=str, default='default')

# Optimization Based Parameters
parser.add_argument('--wdecay', type=float, help="L2 Norm to use")
parser.add_argument('--lr', type=float, default=0.001, help="L2 Norm to use")
parser.add_argument('--opt_alg', type=str, default="adam", help="Optimization algorithm to use")
parser.add_argument('--patience', type=int, default=10, help="Number of layers")


parser.add_argument('--e_dim', type=int, default=128, help="Embedding Dimension")
parser.add_argument('--h_dim', type=int, default=1150, help="Hidden Dimension")
parser.add_argument('--s_dir', help="Serialization directory")
parser.add_argument('--transformer', help="Use transformer unit",action='store_true')
parser.add_argument('--train_embeddings', help="Enable fine-tunning of embeddings like elmo",action='store_true')
parser.add_argument('--IDA', help="Use IDA Encoder",action='store_true')
parser.add_argument('--hashed', help="Use Hashed Memory Networks",action='store_true')
parser.add_argument('--ewc', help="Use Elastic Weight consolidation",action='store_true')
parser.add_argument('--ewc_importance', type=int, default=1000, help="Use Elastic Weight consolidation importance to add weights")
parser.add_argument('--ewc_normalise', type=str, help="Use Elastic Weight consolidation length, batches, none")
parser.add_argument('--task_embed', action='store_true', help="Use the task encoding to encode task id")
parser.add_argument('--position_embed', action='store_true', help="Add the positional embeddings in the word embeddings.")

## Memory related options
parser.add_argument('--mem_size', help="Memory key size", type=int, default=300)
parser.add_argument('--mem_context_size', help="Memory output size", type=int, default=512)
parser.add_argument('--use_memory', action='store_true', help="Weather to use memory are not")
parser.add_argument('--use_binary', action='store_true', help="Make the memory access binary")
parser.add_argument('--pad_memory', action='store_true', help="Pad the Memory after training each task")


parser.add_argument('--inv_temp', help="Inverse temp to use for IDA or other algorithms",type=float, default=None)
parser.add_argument('--temp_inc', help="Increment in temperature after each task",type=float, default=None)
parser.add_argument('--majority', help="Use Sequence to sequence",action='store_true')
parser.add_argument('--tryno', type=int, default=1, help="This is ith try add this to name of df")
parser.add_argument('--small', type=int, default=None, help="Use only these examples from each set")
parser.add_argument('--run_name', type=str, default="Default", help="This is the run name being saved to tensorboard")
parser.add_argument('--storage_prefix', type=str, default="./runs/", help="This is used to store the runs inside runs folder")

parser.add_argument('--pooling', type=str, default="max", help="Selects the pooling operation for CNN, max pooling, min pooling, average pooling. max,min,avg")
parser.add_argument('--no_save_weight', action='store_true', help="Disable saving of weights")

args = parser.parse_args()


#writer=SummaryWriter(run_name)
#writer.add_text("args", str(args))


print("Arguments", args)
tasks = list(args.task)

train_data = {}
dev_data = {}
few_data = {}
vocabulary = {}

for task in tasks:
  utils.load_dataset(task, train_data, dev_data, few_data, args.embeddings, special=True)
  if args.small:
    train_data[task] = train_data[task][:args.small]

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
run_name=args.storage_prefix + args.run_name+"_"+str(args.layers)+"_hdim_"+str(args.h_dim)+"_stride_"+str(args.stride)+"_ngram_"+str(args.ngram_filter)+"_code_"+task_code+"/run_"+str(args.tryno)

vocab = Vocabulary.from_instances(joint_train + joint_dev)

vocab.print_statistics()

word_embeddings = utils.get_embedder(args.embeddings, vocab, args.e_dim, rq_grad=args.train_embeddings)

word_embedding_dim = word_embeddings.get_output_dim()
if args.task_embed:
    word_embedding_dim += 1

model, experiment = net.get_model(vocab, word_embeddings, word_embedding_dim, args)
print("Running Experiment " , experiment)

if not args.no_save_weight:
    save_weight = SaveWeights(experiment, args.layers, args.h_dim, task_code, labels_mapping, args.mean_classifier, tasks=train)

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
                  validation_metric="-loss",
                  num_serialized_models_to_keep=1,
                  model_save_interval=1,
                  serialization_dir=run_name,
                  histogram_interval=100,
                  patience=args.patience,
                  num_epochs=args.epochs,
                  cuda_device=devicea)
  old_task_data = None
  for tid,i in enumerate(train,1):
    print("\nTraining task ", i)
    sys.stdout.flush()
    if args.diff_class:
      if args.pad_memory:
          model.encoder.add_target_pad()
      training_ = True if i != 1 else False
      normaliser = len(train_data[i])/args.bs
      if args.ewc_normalise == 'length':
          normaliser = len(train_data[i])
      elif args.ewc_normalise == 'batches':
          normaliser = len(train_data[i])/args.bs
      else:
          normaliser  = 1
      model.set_task(i, training=training_, normaliser=normaliser)
      trainer._num_epochs = args.epochs
      iterator.index_with(vocabulary[i])
      trainer.train_data = train_data[i]
      trainer._validation_data = dev_data[i]
      trainer.model = model
      trainer.iterator = iterator
      trainer._validation_iterator = iterator
      if i == 'cola':
          trainer._validation_metric = 'average'
          trainer._metric_tracker._should_decrease = False
          trainer.validation_metric = '+average'
      else:
          trainer._validation_metric = 'loss'
          trainer._metric_tracker._should_decrease = True
          trainer.validation_metric = '-loss'
      trainer._metric_tracker.clear()
    if not args.majority:
      metrics = trainer.train()
      trainer._tensorboard.add_train_scalar("restore_checkpoint/"+str(i), metrics['training_epochs'], timestep=tid)
    old_task_data = train_data[i]
    for j in evaluate_tasks:
      print("\nEvaluating ", j)
      sys.stdout.flush()
      if args.diff_class:
        model.set_task(j)
        # This batch size of 10000 is hack to get activation while doing evaluation.
        iterator1 = BucketIterator(batch_size=args.bs, sorting_keys=[("tokens", "num_tokens")])
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
          trainer.model = model
          trainer.train_data = few_data[j]
          trainer._validation_data = few_data[j]
          trainer.iterator = iterator1
          trainer._metric_tracker.clear()
          print("Doing few shot traing current things is ", trainer._metric_tracker._epochs_with_no_improvement, trainer._metric_tracker._epoch_number)
          trainer._num_epochs = 10
          trainer.train()
          # Back to hack of 10000 to get all the activations together as
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
      print("Now evaluating ", j, len(dev_data[j]))
      metric = evaluate(model=model,
	 instances=dev_data[j],
	 data_iterator=iterator1,
	 cuda_device=devicea,
	 batch_weight_key=None)

      # Take first 500 instances for evaluating activations.
      if not args.no_save_weight:
         iterator1 = BucketIterator(batch_size=500, sorting_keys=[("tokens", "num_tokens")])
         iterator1.index_with(vocabulary[j])
         evaluate(model=model,
	     instances=dev_data[j][:500],
	     data_iterator=iterator1,
	     cuda_device=devicea,
	     batch_weight_key=None)
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
  # Calculate the catastrophic forgetting and add it into tensorboard before
  # closing the tensorboard
  c_standard_metric = get_catastrophic_metric(train, ostandard_metrics)
  print("Forgetting Results", c_standard_metric)
  for tid,task in enumerate(c_standard_metric, 1):
      trainer._tensorboard.add_train_scalar("forgetting_metric/standard_" + task,
					    c_standard_metric[task],
					    timestep=tid)
  if not args.no_save_weight:
      #save_weight.write_activations(overall_metrics, trainer, tasks)
      save_weight.get_task_tsne(trainer)

  trainer._tensorboard._train_log.close()

if not args.diff_class:
  print("\n Joint Evaluating ")
  sys.stdout.flush()
  overall_metric = evaluate(model=model,
	 instances=joint_dev,
	 data_iterator=iterator,
	 cuda_device=devicea,
	 batch_weight_key=None)

print("Training Results are on these Arguments", args)

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
