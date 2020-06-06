import torch.nn as nn
import torch
import math
from allennlp.nn.util import move_to_device

class TaskProjection(nn.Module):

    def __init__(self, hdim, dropout=0.1, max_tasks=20):
        super(TaskProjection, self).__init__()
        self.dropouti = nn.Dropout(p=dropout)
        self.hdim = hdim
        self.max_tasks = max_tasks
        self.dropout = dropout
        self.linear_layer = nn.Linear(max_tasks, hdim)

    def forward(self, x, task_id):
        task_input = torch.zeros(self.max_tasks)
        task_input[task_id] = 1
        if torch.cuda.is_available():
            task_input = move_to_device(task_input, torch.cuda.current_device())
        task_projection = self.linear_layer(task_input)
        #print("X", x, x.shape)
        #print("Task projection", task_projection, task_projection.shape)
        #print(" Sum : ", x+task_projection, (x+task_projection).shape)
        x = x + task_projection
        return x
