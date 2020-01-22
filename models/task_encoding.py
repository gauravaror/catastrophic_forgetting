import torch.nn as nn
import torch
import math

class TaskEncoding(nn.Module):

    def __init__(self, hdim, dropout=0.1, max_tasks=20):
        super(TaskEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        te = torch.zeros(max_tasks, hdim)
        position = torch.arange(0, max_tasks, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hdim, 2).float() * (-math.log(10000.0) / hdim))
        te[:, 0::2] = torch.sin(position * div_term)
        te[:, 1::2] = torch.cos(position * div_term)
        te = te.unsqueeze(0)
        self.register_buffer('te', te)

    def forward(self, x, task_id):
        x = x + self.te[: ,task_id, :]
        return self.dropout(x)
