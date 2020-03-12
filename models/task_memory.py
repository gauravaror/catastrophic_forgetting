import torch
import torch.nn as nn

class TaskMemory(nn.Module):
    def __init__(self, emb_dim, memory_size):
        super(TaskMemory, self).__init__()
        self.emb_size  = emb_dim
        self.memory_size  = memory_size
        self.current_index = 0
        self.task_to_mem_id = {}
        self.mem_id_to_task = {}
        self.memory_module = nn.ModuleList()
        self.loss_function = nn.MSELoss(reduction='mean')

    def add_task_memory(self, task_id):
        self.task_to_mem_id[task_id] = self.current_index
        self.mem_id_to_task[self.current_index] = task_id
        self.memory_module.append(nn.Linear(self.emb_size, self.memory_size))
        self.current_index += 1

    def forward(self, input_embeddings, task):
        task_index = self.task_to_mem_id[task]
        self.mem = {}
        index_upto = self.task_to_mem_id[task] + 1
        for tid in range(index_upto):
            task_identifier = self.mem_id_to_task[tid]
            self.mem[task_identifier] = self.memory_module[tid](input_embeddings)
        output = torch.cat([input_embeddings, self.mem[task]], dim=input_embeddings.dim()-1)
        return output

    def get_input_size(self):
        return self.emb_size

    def get_output_dim(self):
        return self.emb_size + self.memory_size

    def get_memory_loss(self, task):
        index_upto = self.task_to_mem_id[task]
        curr_memory = self.mem[task]
        loss = 0
        for i in range(index_upto):
            this_task_id = self.mem_id_to_task[i]
            mem_con = self.mem[this_task_id]
            loss += self.loss_function(curr_memory, mem_con)
        print("Task memory loss", loss)
        return loss
