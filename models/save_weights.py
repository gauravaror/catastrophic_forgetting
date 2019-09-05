import torch
from models.cnn_encoder import CnnEncoder
import pickle
import pandas as pd
import sys
from models import get_cca as wcca
from models import utils

sys.path.append("..")
from svcca import cca_core as svc
import numpy as np


class SaveWeights:

  def __init__(self, encoder_, layer, hdim, code, labels_mapping, mean):
    self.encoder_type = encoder_
    self.tryno=1
    self.activations={}
    self.labels={}
    self.weights={}
    self.layer=layer
    self.hdim=hdim
    self.code=code
    self.labels_map = labels_mapping

    # Stuff to make mean classifier show up on tsne plots
    self.mean_classifier = mean
    self.mean_representation = {}
    self.encoder_representation = {}

  def add_activations(self, model, train, evaluated):
    if not train in self.activations:
      self.activations[train] = {}
      self.labels[train] = {}
      self.mean_representation[train] = {}
      self.encoder_representation[train] = {}
      if self.encoder_type == "cnn":
          self.weights[train] = self.get_weights(model.encoder)
      if self.encoder_type == "lstm":
          self.weights[train] = self.get_weights(model.encoder._module)
    self.activations[train][evaluated], self.labels[train][evaluated] = model.get_activations()
    if self.mean_classifier:
        self.mean_representation[train][evaluated], self.encoder_representation[train][evaluated] = model.get_mean_representation()

  def get_zero_weights(self, activations):

    # Activations are samplesXneuron activation. Count activation
    # Which are non-zero at axis one. one with zero nonzero are dead
    # Neurons.
    axisz_non=np.count_nonzero(activations, axis=0)
    axiso_non=np.count_nonzero(activations, axis=1)
    average_zero_neurons=sum(axiso_non)/len(axiso_non)
    second_size=activations.shape[1]
    dead_neurons=np.count_nonzero(axisz_non)
    return (len(axisz_non)-dead_neurons),(second_size-average_zero_neurons),second_size
      
  def set_stat(self, evalua, task, metric, metric_value, trainer, val, tasks):
    puttask = evalua
    timeset = (tasks.index(task) + 1)
    print("Adding training scalar: ", metric, " timeset ", timeset,
	  ' evaluate ', evalua, ' task ', task,
          ' metric val ', metric_value)
    if (metric == 'total'):
        puttask=''
        timeset = (tasks.index(task) + 1)
    #elif metric == 'weight_corr' or metric == 'weight_corr_svcc':
        #puttask=''
        #timeset = (tasks.index(evalua) + 1)

    trainer._tensorboard.add_train_scalar("weight_stats/"+metric+"/"+str(puttask)+'/',
            metric_value,
            timestep=timeset)
    val[metric] = metric_value
    return val

  def get_arr_rep(self, data, task):
    # This is used to find the test instances currently being processed.
    test_instances = {'trec': 500, 'sst': 1101, 'subjectivity': 1000, 'cola': 527, 'ag': 1500, 'sst_2c': 872}
    if self.encoder_type == 'cnn':
      new_representation = torch.cat(data, dim=2)
      samples, filters, gram = new_representation.shape
      new_representation =  new_representation.reshape(filters, samples*gram)
      new_representation = utils.torch_remove_zero(new_representation)
      return new_representation
    elif self.encoder_type == 'lstm':
      print("Shape of tensors ", data.shape)
      data = data.to('cpu').detach()
      if data.shape[1] < data.shape[0]:
          data = data.reshape(data.shape[1], data.shape[0])
      data = utils.torch_remove_zero(data)
      return data.numpy()
#.reshape(test_instances[task], -1).numpy()

  def write_activations(self, overall_metrics, trainer, tasks):
    lista={'trec': 500, 'sst': 1101, 'subjectivity': 1000, 'cola': 527, 'ag': 1500, 'sst_2c': 872}
    final_val=[]
    allowed_tasks = []
    for task in self.activations.keys():
      allowed_tasks.append(task)
      for evalua in self.activations[task].keys():
              if not evalua in allowed_tasks:
                continue
              val={}
              # Extract Activations and get correlation.
              # Array representation to get activations in CNN: Filtersxsamples.
              # input must be number of neuronsby datapoints hence adjusting that for lstm.
              # first dimension should be bigger than second. 
              first_activation = self.get_arr_rep(self.activations[evalua][evalua], evalua)
              current_activation = self.get_arr_rep(self.activations[task][evalua], evalua)
              try:
                  corr = svc.get_cca_similarity(first_activation, current_activation)
              except Exception as e:
                  print("task Failed SVC activation ", task, evalua)
                  print(e)
                  corr = {}
                  corr['mean'] = (0,0)
              cor1 = corr['mean'][0]

              this_label = self.labels[task][evalua].cpu().numpy()
              if self.mean_classifier:
                      this_mean = self.mean_representation[task][evalua][evalua]
                      this_encoder = self.encoder_representation[task][evalua].cpu().numpy()
                      plot = utils.run_tsne_embeddings(this_encoder, this_label, task, evalua,
                                                       lay, gram, self.labels_map, this_mean)
              else:
                      plot = utils.run_tsne_embeddings(current_activation, this_label, task,
                                                       evalua, 0, 0, self.labels_map)
              label_figure  = "TSNE_embeddings/" + str(task) + "/"+ evalua
              trainer._tensorboard._train_log.add_image(label_figure, plot, dataformats='NCHW')
              dead, average_z,tot = self.get_zero_weights(current_activation)
              val = self.set_stat(evalua, task, 'avg_zeros_per', average_z/tot, trainer, val, tasks)
              val = self.set_stat(evalua, task, 'dead_per', dead/tot, trainer, val, tasks)
              val = self.set_stat(evalua, task, 'corr', float(cor1), trainer, val, tasks)
              val['total'] = tot

              # Extract Weights
              if len(self.weights) > 0:
                first_weight = torch.cat(self.weights[evalua], dim=1)
                first_weight = first_weight.reshape(first_weight.shape[0], -1)

                current_weight = torch.cat(self.weights[task], dim=1)
                current_weight = current_weight.reshape(current_weight.shape[0], -1)
                if self.encoder_type == 'lstm':
                    if first_weight.shape[0] > first_weight.shape[1]:
                        first_weight = first_weight.reshape(first_weight.shape[1], first_weight.shape[0])
                        current_weight = current_weight.reshape(current_weight.shape[1], current_weight.shape[0])
                current_weight = utils.torch_remove_zero(current_weight)
                first_weight = utils.torch_remove_zero(first_weight)
                try:
                    weight_corr_svc = svc.get_cca_similarity(first_weight, current_weight)
                    weight_corr = weight_corr_svc['mean'][0]
                except Exception as e:
                    print("task Failed SVC weight ", task, evalua)
                    print(e)
                    weight_corr = 0
              else:
                weight_corr = 'nan'

              if weight_corr != 'nan':
                  val = self.set_stat(evalua, task, 'weight_corr_svcc', float(weight_corr), trainer, val, tasks)
              val['evaluate']=str(evalua)
              val['task']=str(task)
              val['layer'] = self.layer
              val['h_dim'] = self.hdim
              val['code'] = self.code
              val['accuracy'] = overall_metrics[evalua][task]['accuracy']
              final_val.append(val)
    mydf=pd.DataFrame(final_val)
    filename="final_corr/%s_layer_%s_hdim_%s_code_%s.df"%(str(self.encoder_type),str(self.layer),str(self.hdim), str(self.code))
    mydf.to_pickle(filename)

    
  def get_weights(self, model):
      model_weights = []
      for name, param in model.named_parameters():
          if 'weight' in name:
              ww = param.to('cpu').detach()
              model_weights.append(ww)
              print("Got model weights : ", name, " with  shape", ww.shape)
      return model_weights
