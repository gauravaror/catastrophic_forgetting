import sys
from allennlp.training.util import evaluate
from allennlp.nn.util import move_to_device
from allennlp.data.iterators import BucketIterator
import torch
import pandas as pd
import torch.nn as nn
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

    def forward(self, inputs, labels):
        x = self.encoder(inputs)
        x = self.activation(x)
        tag_logits = self.classification(x)
        loss = self.loss_function(tag_logits, labels)
        return loss, tag_logits


def task_diagnostics(tasks, train_data, val_data, vocabulary, model, args):
    devicea = -1
    if torch.cuda.is_available():
        devicea = 0
    train_activations = None
    test_activations = None
    train_labels = None
    test_labels = None
    for tid,task in enumerate(tasks):
            model.set_task(task)
            iterator1 = BucketIterator(batch_size=500, sorting_keys=[("tokens", "num_tokens")])
            iterator1.index_with(vocabulary[task])
            evaluate(model=model,
                     instances=train_data[task][:1000],
                     data_iterator=iterator1,
                     cuda_device=devicea,
                     batch_weight_key=None)
            train_act, _ = model.get_activations()
            train_lab = torch.LongTensor(train_act.size(0)).fill_(tid)
            evaluate(model=model,
                     instances=val_data[task][:500],
                     data_iterator=iterator1,
                     cuda_device=devicea,
                     batch_weight_key=None)
            test_act, _ = model.get_activations()
            test_lab = torch.LongTensor(test_act.size(0)).fill_(tid)
            if train_activations is None or test_activations is None:
                train_activations = train_act
                test_activations = test_act
                train_labels = train_lab
                test_labels = test_lab
            else:
                train_activations = torch.cat([train_activations, train_act], dim=0)
                test_activations = torch.cat([test_activations, test_act], dim=0)
                train_labels = torch.cat([train_labels, train_lab], dim=0)
                test_labels = torch.cat([test_labels, test_lab], dim=0)
            print("Activations ", train_activations.shape, test_activations.shape, train_labels.shape, test_labels.shape)

    train_activations = move_to_device(train_activations, devicea)
    test_activations = move_to_device(test_activations, devicea)
    train_labels = move_to_device(train_labels, devicea)
    test_labels = move_to_device(test_labels, devicea)

    diag_model = DiagnositicClassifier(train_activations.size(1), 128, len(tasks))
    diag_model = diag_model.cuda(devicea)
    optimizer = utils.get_optimizer(args.opt_alg, diag_model.parameters(), args.lr, args.wdecay)
    for epoch in range(100):
        diag_model.train()
        optimizer.zero_grad()
        loss, _ = diag_model(train_activations, train_labels)
        loss.backward()
        optimizer.step()

    vloss , logits = diag_model(test_activations, test_labels)
    _, predicted = torch.max(logits, 1)
    correct_ones = (predicted == test_labels).sum()
    print("Validation Loss", "%s/%s"%(correct_ones.item(), len(logits)),"vloss", vloss, predicted, test_labels)
    return correct_ones.item(), len(logits)



