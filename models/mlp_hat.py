import sys
import torch

import utils

class MLPHat(torch.nn.Module):

    def __init__(self,inputsize,taskcla,nlayers=2,nhid=2000,pdrop1=0.2,pdrop2=0.5):
        super(MLPHat,self).__init__()

        size=inputsize
        self.taskcla=taskcla

        self.nlayers=nlayers
        self.nhid = nhid

        self.relu=torch.nn.ReLU()
        self.fc1=torch.nn.Linear(size,nhid)
        self.efc1=torch.nn.Embedding(len(self.taskcla),nhid)
        if nlayers>1:
            self.fc2=torch.nn.Linear(nhid,nhid)
            self.efc2=torch.nn.Embedding(len(self.taskcla),nhid)
            if nlayers>2:
                self.fc3=torch.nn.Linear(nhid,nhid)
                self.efc3=torch.nn.Embedding(len(self.taskcla),nhid)
        self.last=torch.nn.ModuleList()

        self.gate=torch.nn.Sigmoid()
        """ (e.g., used with compression experiments)
        lo,hi=0,2
        self.efc1.weight.data.uniform_(lo,hi)
        self.efc2.weight.data.uniform_(lo,hi)
        self.efc3.weight.data.uniform_(lo,hi)
        #"""

        return

    def get_output_dim(self):
        return self.nhid

    def forward(self,t,x,s=1):
        # Gates
        masks=self.mask(t,s=s)
        if self.nlayers==1:
            gfc1=masks
        elif self.nlayers==2:
            gfc1,gfc2=masks
        elif self.nlayers==3:
            gfc1,gfc2,gfc3=masks
        # Gated
        h = x
        h = self.relu(self.fc1(h))
        h=h*gfc1.expand_as(h)
        if self.nlayers>1:
            h = self.relu(self.fc2(h))
            h=h*gfc2.expand_as(h)
            if self.nlayers>2:
                h= self.relu(self.fc3(h))
                h=h*gfc3.expand_as(h)
        output = h.mean(dim=1).squeeze(1)
        return output,masks

    def mask(self,t,s=1):
        gfc1=self.gate(s*self.efc1(t))
        if self.nlayers==1: return gfc1
        gfc2=self.gate(s*self.efc2(t))
        if self.nlayers==2: return [gfc1,gfc2]
        gfc3=self.gate(s*self.efc3(t))
        return [gfc1,gfc2,gfc3]

    def get_view_for(self,n,masks):
        if self.nlayers==1:
            gfc1=masks
        elif self.nlayers==2:
            gfc1,gfc2=masks
        elif self.nlayers==3:
            gfc1,gfc2,gfc3=masks
        if n=='fc1.weight':
            return gfc1.data.view(-1,1).expand_as(self.fc1.weight)
        elif n=='fc1.bias':
            return gfc1.data.view(-1)
        elif n=='fc2.weight':
            post=gfc2.data.view(-1,1).expand_as(self.fc2.weight)
            pre=gfc1.data.view(1,-1).expand_as(self.fc2.weight)
            return torch.min(post,pre)
        elif n=='fc2.bias':
            return gfc2.data.view(-1)
        elif n=='fc3.weight':
            post=gfc3.data.view(-1,1).expand_as(self.fc3.weight)
            pre=gfc2.data.view(1,-1).expand_as(self.fc3.weight)
            return torch.min(post,pre)
        elif n=='fc3.bias':
            return gfc3.data.view(-1)
        return None

