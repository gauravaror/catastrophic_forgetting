from typing import Iterator, List, Dict
import torch
import torch.optim as optim
import numpy as np
import sys
import copy
import pandas as pd
import random
from collections import defaultdict

from models.utils import get_catastrophic_metric
from allennlp.data.vocabulary import Vocabulary
from models.save_weights import SaveWeights

from allennlp.training.metrics import CategoricalAccuracy
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.util import move_optimizer_to_cuda, evaluate
from allennlp.common.params import Params
from allennlp.nn.util import move_to_device

import models.net as net
from models.args import get_args
import models.utils as utils
import models.evaluate as eva
import models.task_diagnostics as diag
#from torch.utils.tensorboard import SummaryWriter

from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.handlers import EarlyStopping, Checkpoint, DiskSaver, global_step_from_engine

from tensorboardX import SummaryWriter

torch.autograd.set_detect_anomaly(True)

args = get_args()

#writer=SummaryWriter(run_name)
#writer.add_text("args", str(args))


print("Arguments", args)
tasks = list(args.task)

train_data = {}
dev_data = {}
few_data = {}
vocabulary = {}

smax = 400
lamb = 0.75
#lamb = 0.0001
current_tid = 0
clipgrad = 10000 

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
    if args.transformer and word_embedding_dim % 2 != 0:
        raise Exception("Transformer and task embedded should have even dimension for PositionalEmbeddings")

model, experiment = net.get_model(vocab, word_embeddings, word_embedding_dim, args, tasks)
print("Running Experiment " , experiment)

save_weight = None
if not args.no_save_weight:
    save_weight = SaveWeights(experiment, args.layers, args.h_dim, task_code, labels_mapping, args.mean_classifier, tasks=train)

for i in tasks:
  model.add_task(i, vocabulary[i])
if torch.cuda.is_available():
  model.cuda(0)

optimizer = utils.get_optimizer(args.opt_alg, model.parameters(), args.lr, args.wdecay)

torch.set_num_threads(4)
iterator = BucketIterator(batch_size=args.bs, sorting_keys=[("tokens", "num_tokens")])

writer = SummaryWriter(logdir=run_name)
writer.add_text("args", str(args))

iterator.index_with(vocab)
devicea = -1
if torch.cuda.is_available():
  devicea = 0

overall_metrics = {}
ostandard_metrics = {}

mask_pre = None
mask_back = None

def criterion(loss, masks):
    reg=0
    count=0
    if mask_pre is not None:
        for m,mp in zip(masks, mask_pre):
            aux = 1-mp
            reg += (m*aux).sum()
            count += aux.sum()
    else:
        for m in masks:
            reg += m.sum()
            count += np.prod(m.size()).item()
    reg /= count
    return loss + lamb*reg


def update(engine, batch):
    model.train()
    optimizer.zero_grad()
    batch  = move_to_device(batch, devicea)
    output = model(batch['tokens'], batch['label'])
    output["loss"].backward()
    optimizer.step()
    return output

def update_hat(engine, batch):
    thres_cosh=50
    thres_emb=6
    model.train()
    optimizer.zero_grad()
    s=(smax-1/smax)*engine.state.iteration/engine.state.epoch_length+1/smax
    batch  = move_to_device(batch, devicea)
    output = model(batch['tokens'], batch['label'], s=s)
    mask = model.get_hat_masks()
    #task= torch.LongTensor([tid]).cuda()
    #mask = copy.copy(model.encoder.mask(task, s))
    #mask = copy.copy(mask)
    output["loss"] = criterion(output["loss"], mask)
    output["loss"].backward()
    # Restrict layer gradients in backprop
    if current_tid > 1:
        for n,p in model.encoder.named_parameters():
            if n in mask_back:
                p.grad.data *= mask_back[n]

    # Compensate embedding gradients
    for n,p in model.encoder.named_parameters():
        if n.startswith('e'):
            num = torch.cosh(torch.clamp(s*p.data,-thres_cosh,thres_cosh)) + 1
            den = torch.cosh(p.data) + 1
            p.grad.data *= smax/s*num/den

    # Apply step
    #torch.nn.utils.clip_grad_norm(model.parameters(), clipgrad)
    optimizer.step()
    return output


def validate(engine, batch):
    model.eval()
    batch = move_to_device(batch, devicea)
    model.get_metrics()
    output = model(batch['tokens'], batch['label'])
    current_metric = model.get_metrics()
    engine.state.metric = current_metric
    engine.state.metric['loss'] = output['loss']


if args.mlp_hat:
    itrainer = Engine(update_hat)
else:
    itrainer = Engine(update)
ievaluator = Engine(validate)


@itrainer.on(Events.EPOCH_COMPLETED(every=5))
def log_training(engine):
    batch_loss = engine.state.output['loss']
    ewc_loss = 0
    if 'ewc_loss' in engine.state.output:
        ewc_loss = engine.state.output['ewc_loss']
    metric = model.get_metrics()
    lr = optimizer.param_groups[0]['lr']
    e = engine.state.epoch
    n = engine.state.max_epochs
    i = engine.state.iteration
    print("Epoch {}/{} : {} - batch loss: {}, ewc loss: {}, lr: {}, accuracy: {}, average: {} ".format(e, n, i, batch_loss, ewc_loss, lr, metric['accuracy'], metric['average']))

pbar = ProgressBar()
pbar.attach(itrainer, ['loss'])
current_task = None

@itrainer.on(Events.EPOCH_COMPLETED)
def run_validation(engine):
    val_iterator = BucketIterator(batch_size=args.bs, sorting_keys=[("tokens", "num_tokens")])
    val_iterator.index_with(vocabulary[current_task])
    raw_val_generator = iterator(dev_data[current_task], num_epochs=1)
    val_groups = list(raw_val_generator)
    model.get_metrics(True)
    ievaluator.run(val_groups)
    batch_loss = ievaluator.state.metric['loss']
    metric = ievaluator.state.metric
    lr = optimizer.param_groups[0]['lr']
    e = engine.state.epoch
    n = engine.state.max_epochs
    i = engine.state.iteration
    print("Val Epoch {}/{} : {} - batch loss: {}, lr: {}, accuracy: {}, average: {} ".format(e, n, early_stop_metric.counter, batch_loss, lr, metric['accuracy'], metric['average']))


def score_function(engine):
    metric = engine.state.metric['accuracy']
    if current_task == 'cola':
        metric = engine.state.metric['average']
    return metric

early_stop_metric = EarlyStopping(patience=args.patience, score_function=score_function, trainer=itrainer)
ievaluator.add_event_handler(Events.COMPLETED, early_stop_metric)

to_save = {'model': model}
disk_saver = DiskSaver(run_name, create_dir=True, require_empty=args.require_empty)
best_save = Checkpoint(to_save,
                       disk_saver,
                       n_saved=1,
                       filename_prefix='best',
                       score_function=score_function,
                       score_name="val_best",
                       global_step_transform=global_step_from_engine(itrainer))
ievaluator.add_event_handler(Events.COMPLETED, best_save)

def reset_state(reset_model=True):
    if reset_model:
        print("Best of last task", best_save.last_checkpoint)
        best_save.load_objects({'model': model}, {'model': torch.load(run_name + "/" + best_save.last_checkpoint)})
        best_save._saved = []
    early_stop_metric.counter = 0
    early_stop_metric.best_score = None
    itrainer.state.epoch = 0

# Temperature set
temps = defaultdict(lambda: 1)
if args.inv_temp:
    temps[train[0]] = args.inv_temp
    temp_temp = args.inv_temp
    for i in train[1:]:
        temp_temp *= args.temp_inc
        temps[i] = temp_temp
    print("temperatures startt", temps)

for tid,i in enumerate(train,1):
    current_task = i
    current_tid = tid
    print("\nTraining task ", i)
    sys.stdout.flush()
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
    if args.ewc:
        normaliser_ = normaliser
    else:
        normaliser_ = None
    model.set_task(i, training=training_, normaliser=normaliser_, tmp=temps[i])
    iterator = BucketIterator(batch_size=args.bs, sorting_keys=[("tokens", "num_tokens")])
    iterator.index_with(vocabulary[i])
    raw_train_generator = iterator(train_data[i], num_epochs=1)
    groups = list(raw_train_generator)
    itrainer.run(groups, max_epochs=args.epochs)
    if args.oewc:
        reset_state(reset_model=False)
        model.set_task(i, training=training_, normaliser=normaliser, tmp=temps[i])
        raw_train_generator = iterator(train_data[i][:args.ewc_samples], num_epochs=1)
        groups = list(raw_train_generator)
        itrainer.run(groups, max_epochs=1)
        reset_state()
    else:
        reset_state()

    if args.mlp_hat:
        with torch.no_grad():
            # Activations mask
            #task = torch.autograd.Variable(torch.LongTensor([current_tid]).cuda(),volatile=False)
            #mask = model.encoder.mask(task, smax)
            mask = copy.copy(model.get_hat_masks())

            if current_tid == 1:
                mask_pre = mask
            else:
                for ind in range(len(mask_pre)):
                    mask_pre[ind] = torch.max(mask_pre[ind], mask[ind])

            # Weights mask
            mask_back={}
            for na,_ in model.named_parameters():
                vals = model.encoder.get_view_for(na, mask_pre)
                if vals is not None:
                    mask_back[na] = 1 - vals


    ometric, smetric = eva.evaluate_all_tasks(i, evaluate_tasks, dev_data, vocabulary,
                                                             model, args, save_weight, temps)
    overall_metrics[i] = ometric
    ostandard_metrics[i] = smetric
    count = 0
    metric_acc = 0
    micro_avg_acc = 0
    std_evl_acc = 0
    for j in smetric.keys():
        writer.add_scalar("evaluate/"+str(j),
                                              float(ometric[j]['metric']),
                                              tid)
        writer.add_scalar("standard_evaluate/"+str(j),
                                              smetric[j],
                                              tid)
        writer.add_scalar("micro_avg/"+str(j),
                                              float(ometric[j]['micro_avg']),
                                              tid)
        count += 1
        metric_acc += float(ometric[j]['metric'])
        std_evl_acc += float(smetric[j])
        micro_avg_acc += float(ometric[j]['micro_avg'])
    if count == len(train):
        print("Average-Accuracy ",task_code,i, 'Metric', metric_acc/count, 'standard', std_evl_acc/count, 'micro', micro_avg_acc/count)
        writer.add_scalar("avg_accuracy/evaluate", metric_acc/count)
        writer.add_scalar("avg_accuracy/standard_evaluate", std_evl_acc/count)
        writer.add_scalar("avg_accuracy/micro_avg", micro_avg_acc/count)


diag_metrics = diag.task_diagnostics(tasks, train_data, dev_data, vocabulary, model, args)
for task,dacc in diag_metrics.items():
    print("Diag accur tensorboard ", task, dacc)
    writer.add_scalar('task_diagnostics/{}'.format(task), dacc)



# Calculate the catastrophic forgetting and add it into tensorboard before
# closing the tensorboard
c_standard_metric = get_catastrophic_metric(train, ostandard_metrics)
print("Forgetting Results", c_standard_metric)
for tid,task in enumerate(c_standard_metric, 1):
  writer.add_scalar("forgetting_metric/standard_" + task,
                                        c_standard_metric[task],
                                        tid)
if not args.no_save_weight:
  #save_weight.write_activations(overall_metrics, writer, tasks)
  save_weight.get_task_tsne(writer)

writer.close()

print("Training Results are on these Arguments", args)
eva.print_evaluate_stats(train, evaluate_tasks, args, overall_metrics, task_code, experiment)

