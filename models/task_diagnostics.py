import sys
import pandas as pd

from allennlp.training.util import evaluate
from allennlp.nn.util import move_to_device
from allennlp.data.iterators import BucketIterator

import torch
import torch.nn.functional as F
import torch.nn as nn

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.handlers import EarlyStopping

import models.utils as utils
class DiagnositicClassifier(nn.Module):
    def __init__(self, input_dim: int,
                       h_dim: int,
                       num_labels: int):
        super(DiagnositicClassifier, self).__init__()
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.num_labels = num_labels
        self.encoder = nn.Linear(input_dim, h_dim)
        self.activation = nn.ReLU()
        self.classification = nn.Linear(h_dim, num_labels)
        self.loss_function = torch.nn.CrossEntropyLoss()

    def forward(self, data):
        inputs = data
        #labels = data[1]
        x = self.encoder(inputs)
        x = self.activation(x)
        tag_logits = self.classification(x)
        #loss = self.loss_function(tag_logits, labels)
        output = F.log_softmax(tag_logits, dim=-1)
        return output


def evaluate_get_dataset(model, task, vocab, dataset, num_samples, task_id):
    devicea = -1
    if torch.cuda.is_available():
        devicea = 0
    iterator1 = BucketIterator(batch_size=500, sorting_keys=[("tokens", "num_tokens")])
    iterator1.index_with(vocab)
    model.set_task(task)
    evaluate(model=model,
             instances=dataset[:num_samples],
             data_iterator=iterator1,
             cuda_device=devicea,
             batch_weight_key=None)
    train_act, _ = model.get_activations()
    if type(train_act) == list:
        # Hack for CNN need to do better
        train_act = train_act[-1]
        print("tran ", train_act.shape)
        train_act = train_act.reshape(train_act.size(0), -1)
        train_act = train_act[:, :128]
    train_lab = torch.LongTensor(train_act.size(0)).fill_(task_id)

    return move_to_device(train_act, devicea) , move_to_device(train_lab, devicea)

def task_diagnostics(tasks, train_data, val_data, vocabulary, model, args):
    devicea = -1
    if torch.cuda.is_available():
        devicea = 0
    train_activations = []
    test_activations = []
    train_labels = []
    test_labels = []

    for tid,task in enumerate(tasks):
        train_act, train_lab = evaluate_get_dataset(model, task, vocabulary[task],
                                                   train_data[task], 1000, tid)
        test_act, test_lab = evaluate_get_dataset(model, task, vocabulary[task],
                                                  val_data[task], 500, tid)
        train_activations.append(train_act)
        test_activations.append(test_act)
        train_labels.append(train_lab)
        test_labels.append(test_lab)

    train_activations = torch.cat(train_activations, dim=0)
    test_activations = torch.cat(test_activations, dim=0)
    train_labels = torch.cat(train_labels, dim=0)
    test_labels = torch.cat(test_labels, dim=0)
    print("Activations ", train_activations.shape, test_activations.shape, train_labels.shape, test_labels.shape)
    # Datasets
    train_ds = torch.utils.data.TensorDataset(train_activations, train_labels)
    test_ds = torch.utils.data.TensorDataset(test_activations, test_labels)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=32)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=2100)

    # Models and Optimizer
    diag_model = DiagnositicClassifier(train_activations.size(1), 128, len(tasks))
    if devicea != -1:
        diag_model.cuda(devicea)
    optimizer = utils.get_optimizer(args.opt_alg, diag_model.parameters(), args.lr, args.wdecay)
    criterion = nn.CrossEntropyLoss()

    # ignite training loops
    if devicea == -1:
        trainer = create_supervised_trainer(diag_model, optimizer, criterion)
        evaluator = create_supervised_evaluator(diag_model, {"accuracy": Accuracy(), "loss": Loss(criterion)})
        val_evaluator = create_supervised_evaluator(diag_model, {"accuracy": Accuracy(), "loss": Loss(criterion)})
    else:
        trainer = create_supervised_trainer(diag_model, optimizer, diag_model.loss_function, device=devicea)
        evaluator = create_supervised_evaluator(diag_model, metrics={'accuracy': Accuracy()}, device=devicea)
        val_evaluator = create_supervised_evaluator(diag_model, metrics={'accuracy': Accuracy()}, device=devicea)
    @trainer.on(Events.EPOCH_COMPLETED)
    def compute_metrics(engine):
        evaluator.run(train_dl)
        print("Epoch", engine.state.epoch, "Training Accuracy", evaluator.state.metrics["accuracy"])
        val_evaluator.run(test_dl)
        print("Validation Accuracy", val_evaluator.state.metrics["accuracy"])

    def score_function(engine):
        return engine.state.metrics['accuracy']
    
    early_stop_metric = EarlyStopping(patience=20, score_function=score_function, trainer=trainer)
    val_evaluator.add_event_handler(Events.COMPLETED, early_stop_metric)
    trainer.run(train_dl, max_epochs=1000)
    logits, test_labels = val_evaluator.state.output

    _, predicted = torch.max(logits, 1)
    correct_ones = (predicted == test_labels).sum()
    metrics = {}
    for i,task in enumerate(tasks):
        start = i*500
        end = (i+1)*500
        correct_this = (predicted[start:end] == test_labels[start:end]).sum()
        metrics[task] = correct_this.item()/500
        #print("Task based accuracy", start, end , task, correct_this)

    metrics["overall"] = val_evaluator.state.metrics["accuracy"]
    print(metrics)
    return metrics
