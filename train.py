from typing import Iterator, List, Dict
import torch
import torch.optim as optim
import numpy as np
import sys
import copy
import pandas as pd
import random
from collections import defaultdict

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset

from models.utils import get_catastrophic_metric
from allennlp.data.vocabulary import Vocabulary
from models.save_weights import SaveWeights
from models.mnist2 import get as mnist2
from models.cifar import get as cifar

from allennlp.training.metrics import CategoricalAccuracy
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.util import move_optimizer_to_cuda, evaluate
from allennlp.common.params import Params
from allennlp.nn.util import move_to_device

import models.net as net
from models.args import get_args
from models.mlp_hat import Net
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


smax = 400
lamb = 0.75
current_tid = 0
clipgrad = 10000

#dataset, train, sizes = get_dataset()
dataset, train, sizes = cifar()

model = Net(sizes, train, 1, 900)  


#for i in tasks:
#  model.add_task(i, vocabulary[i])
if torch.cuda.is_available():
  model.cuda(0)

optimizer = utils.get_optimizer(args.opt_alg, model.parameters(), args.lr, args.wdecay)

torch.set_num_threads(4)
#iterator = BucketIterator(batch_size=args.bs)

devicea = -1
if torch.cuda.is_available():
  devicea = 0

overall_metrics = {}
ostandard_metrics = {}

mask_pre = None
mask_back = None

metric = Accuracy()
ce = torch.nn.CrossEntropyLoss()

def criterion(masks, outputs, target):
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
    reg/=count
    return ce(outputs, target) + lamb*reg

def update_hat(engine, batch):
    thres_cosh=50
    thres_emb=6
    model.train()
    optimizer.zero_grad()
    s=(smax-1/smax)*engine.state.iteration/engine.state.epoch_length+1/smax
    batch  = move_to_device(batch, devicea)
    images = batch[0]
    targets = batch[1]
    ta = torch.LongTensor([current_tid])
    ta  = move_to_device(ta, devicea)
    output,_ = model(ta, images, s)
    loss = criterion(model.mask(ta, s) , output[current_tid], targets)
    loss.backward()
    # Restrict layer gradients in backprop
    if current_tid > 1:
        for n,p in model.named_parameters():
            if n in mask_back:
                p.grad.data *= mask_back[n]

    # Compensate embedding gradients
    for n,p in model.named_parameters():
        if n.startswith('e'):
            num = torch.cosh(torch.clamp(s*p.data,-thres_cosh,thres_cosh)) + 1
            den = torch.cosh(p.data) + 1
            p.grad.data *= smax/s*num/den

    # Apply step
    torch.nn.utils.clip_grad_norm(model.parameters(), clipgrad)
    optimizer.step()
    return output[current_tid], targets


def validate(engine, batch):
    model.eval()
    batch = move_to_device(batch, devicea)
    images = batch[0]
    targets = batch[1]
    ta = torch.LongTensor([current_tid])
    ta  = move_to_device(ta, devicea)
    output,_ = model(ta, images)
    loss = ce(output[current_tid], targets)
    #print(output[current_tid].argmax(dim=1), output[current_tid].shape, targets.shape, targets)
    engine.state.metric = {}
    engine.state.metric['loss'] = loss
    return output[current_tid], targets

itrainer = Engine(update_hat)
ievaluator = Engine(validate)
metric.attach(itrainer, "accuracy")
metric.attach(ievaluator, "accuracy")


@itrainer.on(Events.EPOCH_COMPLETED(every=5))
def log_training(engine):
    batch_loss = 0
    ewc_loss = 0
    lr = optimizer.param_groups[0]['lr']
    e = engine.state.epoch
    n = engine.state.max_epochs
    i = engine.state.iteration
    print("Epoch {}/{} : {} - batch loss: {}, ewc loss: {}, lr: {} ".format(e, n, i, batch_loss, ewc_loss, lr))

pbar = ProgressBar()
pbar.attach(itrainer, ['loss'])
current_task = None

@itrainer.on(Events.EPOCH_COMPLETED)
def run_validation(engine):
    td = TensorDataset(dataset[tid]['test']['x'], dataset[tid]['test']['y'])
    dl = DataLoader(td, batch_size=64)
    ievaluator.run(dl)
    batch_loss = ievaluator.state.metric['loss']
    #metric = ievaluator.state.metric
    lr = optimizer.param_groups[0]['lr']
    e = engine.state.epoch
    n = engine.state.max_epochs
    i = engine.state.iteration
    print("Val Epoch {}/{} : {} - batch loss: {}, lr: {}, accuracy: {} ".format(e, n, early_stop_metric.counter, batch_loss, lr, metric.compute()))


def score_function(engine):
    return metric.compute()

early_stop_metric = EarlyStopping(patience=args.patience, score_function=score_function, trainer=itrainer)
ievaluator.add_event_handler(Events.COMPLETED, early_stop_metric)

to_save = {'model': model}
run_name='runs/mlp_hat_run'
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

for tid,i in enumerate(train):
    current_task = i
    current_tid = tid
    print("\nTraining task ", i)
    sys.stdout.flush()
    if args.pad_memory:
          model.encoder.add_target_pad()
    training_ = True if i != 1 else False
    #model.set_task(i, training=training_, normaliser=normaliser_, tmp=temps[i])
    #iterator = BucketIterator(batch_size=args.bs)
    #iterator.index_with(vocabulary[i])
    #raw_train_generator = iterator(dataset[i]['train'], num_epochs=1)
    #groups = list(raw_train_generator)
    td = TensorDataset(dataset[tid]['train']['x'], dataset[tid]['train']['y'])
    dl = DataLoader(td, batch_size=64)
    itrainer.run(dl, max_epochs=args.epochs)

    if args.mlp_hat:
        with torch.no_grad():
            # Activations mask
            task=torch.autograd.Variable(torch.LongTensor([tid]).cuda(),volatile=False)
            mask = model.mask(task, smax)
            #mask = copy.copy(model.get_masks())
            #mask = model.get_masks()
            for i in range(len(mask)):
                mask[i]=torch.autograd.Variable(mask[i].data.clone(),requires_grad=False)
            mask.required_grad = False

            if current_tid == 0:
                mask_pre = mask
            else:
                mask_pre = torch.max(mask_pre, mask)

            # Weights mask
            mask_back={}
            for n,_ in model.named_parameters():
                vals = model.get_view_for(n, mask_pre)
                if vals is not None:
                    mask_back[n] = 1 - vals


    reset_state()
for tid,i in enumerate(train):
    current_task = i
    current_tid = tid
    print("\n Evaluating task ", i)
    td = TensorDataset(dataset[tid]['test']['x'], dataset[tid]['test']['y'])
    dl = DataLoader(td, batch_size=64)
    ievaluator.run(dl)
    print("Metric ", metric.compute())
