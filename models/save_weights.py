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
          self.weights[train] = self.get_cnn_weights(model)
      #self.activations[train]["trained_task"] = train
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
      
  def set_stat(self, task, evalua, lay, gram, metric, metric_value, trainer, val, tasks):
    puttask = task
    timeset = (tasks.index(evalua) + 1)
    print("Adding training scalar: ", metric, " timeset ", timeset,
	  ' evaluate ', evalua, ' task ', task,
          ' metric val ', metric_value)
    if (metric == 'total'):
        puttask=''
        timeset = (tasks.index(task) + 1)
    elif metric == 'weight_corr':
        puttask=''
        timeset = (tasks.index(evalua) + 1)

    trainer._tensorboard.add_train_scalar("weight_stats/"+metric+"/"+str(puttask)+'/'+str(lay)+'/'+str(gram),
            metric_value,
            timestep=timeset)
    val[metric] = metric_value
    return val

  def write_activations(self, overall_metrics, trainer, tasks):
    lista={'trec': 500, 'sst': 1101, 'subjectivity': 1000, 'cola': 527, 'ag': 1500, 'sst_2c': 872}
    final_val=[]
    first_task=list(self.activations.keys())[0]
    for task in self.activations.keys():
      for evalua in self.activations[task].keys():
            try:
              # Extract Activations
              first_activation=self.activations[first_task][evalua].reshape(lista[evalua],-1).numpy()
              current_activation=self.activations[task][evalua].reshape(lista[evalua],-1).numpy()
              print("Activation Shape", self.activations[task][evalua].shape)
              cor1=svc.get_cca_similarity(first_activation,current_activation)

              this_activation = self.activations[task][evalua][lay][gram].cpu().reshape(lista[evalua],-1).numpy()
              this_label = self.labels[task][evalua].cpu().numpy()
              if self.mean_classifier:
                this_mean = self.mean_representation[task][evalua][evalua]
                this_encoder = self.encoder_representation[task][evalua].cpu().numpy()
                plot = utils.run_tsne_embeddings(this_encoder, this_label, task, evalua, lay, gram, self.labels_map, this_mean)
              else:
                plot = utils.run_tsne_embeddings(this_activation, this_label, task, evalua, lay, gram, self.labels_map)
              label_figure  = "TSNE_embeddings/" + str(task) + "/"+ evalua + "/" + str(lay) + "/" + str(gram)
              trainer._tensorboard._train_log.add_image(label_figure, plot, dataformats='NCHW')

              # Extract Weights
              if len(self.weights) > 0:
                first_weight=self.weights[first_task]
                current_weight=self.weights[task]

                weight_corr=wcca.get_correlation_for_two(first_weight, current_weight)
              else:
                weight_corr='nan'

              val={}
              dead,average_z,tot=self.get_zero_weights(current_activation)
              val = self.set_stat(evalua, task,  'avg_zeros', average_z, trainer, val, tasks)
              val = self.set_stat(evalua, task,  'avg_zeros_per', average_z/tot, trainer, val, tasks)
              val = self.set_stat(evalua, task,  'dead', dead, trainer, val, tasks)
              val = self.set_stat(evalua, task,  'dead_per', dead/tot, trainer, val, tasks)
              val = self.set_stat(evalu, task,  'total', tot, trainer, val, tasks)
              val = self.set_stat(evalu, task,  'corr', float(cor1['mean'][0]), trainer, val, tasks)
              val = self.set_stat(evalu, task,  'weight_corr', float(weight_corr), trainer, val, tasks)
              val['total'] = tot
              val['evaluate']=str(evalua)
              val['task']=str(task)
              val['layer'] = self.layer
              val['h_dim'] = self.hdim
              val['code'] = self.code
              val['accuracy'] = overall_metrics[evalua][task]['accuracy']
              print("task %s Layer %s, gram %s, corr %s"%(str(task),str(evalua),str(gram),str(cor1['mean'])))
            except Exception as e:
              print("task Failed SVC"))
              print(e)
            final_val.append(val)
    mydf=pd.DataFrame(final_val)
    filename="final_corr/%s_layer_%s_hdim_%s_code_%s.df"%(str(self.encoder_type),str(self.layer),str(self.hdim), str(self.code))
    mydf.to_pickle(filename)

    
  def write_weights_new(self, model, layer, h_dim, code, after_task,tryno):
    self.tryno=tryno
    self.write_weights(model, layer, h_dim, code, after_task)
    
  def write_weights(self, model, layer, h_dim, code, after_task):
    if self.encoder_type == "cnn":
      cnn_array=self.get_cnn_weights(model)
      cnn_array['layer'] = layer
      cnn_array['h_dim'] = h_dim
      cnn_array['code'] = code
      cnn_array['after_task'] = after_task
      pickle_out = open("weights/cnn_layer_%s_hdim_%s_CODE_%s_AFTER_%s.weights"%(str(layer),str(h_dim),str(code),str(after_task)), "wb")
      pickle.dump(cnn_array, pickle_out)
      pickle_out.close()

  def get_cnn_weights(self, model):
    cnn_encoder_layer1 = model.encoder._convolution_layers
    cnn_encoder_layer2 = model.encoder._convolution_layers2
    cnn_array={ 0: []}
    for layer in cnn_encoder_layer1:
      this_wei = layer.weight
      this_wei=this_wei.to('cpu')
      curr_array=this_wei.detach().data.numpy()
      cnn_array[0].extend(curr_array)
      print(curr_array.shape)
      this_wei=this_wei.to('cuda')
    for i,layer in enumerate(cnn_encoder_layer2):
      for gram in layer:
        this_wei = gram.weight
        this_wei=this_wei.to('cpu')
        curr_array=this_wei.detach().data.numpy()
        if not ((i+1) in cnn_array):
          cnn_array[(i+1)] = []
        cnn_array[i+1].extend(curr_array)
        print(curr_array.shape)
        this_wei=this_wei.to('cuda')
    return cnn_array
