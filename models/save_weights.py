import torch
from models.cnn_encoder import CnnEncoder
import pickle
import pandas as pd
import sys
sys.path.append("..")
from svcca import cca_core as svc

class SaveWeights:

  def __init__(self, encoder_, layer, hdim, code):
    self.encoder_type = encoder_
    self.tryno=1
    self.activations={}
    self.layer=layer
    self.hdim=hdim
    self.code=code

  def add_activations(self, model, train, evaluated):
    if not train in self.activations:
      self.activations[train] = {}
      #self.activations[train]["trained_task"] = train
    self.activations[train][evaluated] = model.get_activations()

  def write_activations(self):
    lista={'trec': 500, 'sst': 1101, 'subjectivity': 1000, 'cola': 527}
    final_val=[]
    first_task=list(self.activations.keys())[0]
    for task in self.activations.keys():
      for evalua in self.activations[task].keys():
        for lay in range(len(self.activations[task][evalua])):
          for gram in range(len(self.activations[task][evalua][lay])):
            try:
              #print(self.activations[task][task][lay][gram], len(self.activations[task][task][lay][gram]))
              print(self.activations[first_task][evalua][lay][gram].shape, self.activations[task][evalua][lay][gram].shape)
              cor1=svc.get_cca_similarity(self.activations[first_task][evalua][lay][gram].reshape(lista[evalua],-1).numpy(),
                 self.activations[task][evalua][lay][gram].reshape(lista[evalua],-1).numpy())
              val={}
              val['evaluate']=str(evalua)
              val['gram']=str(gram)
              val['lay']=str(lay)
              val['task']=str(task)
              val['corr']=str(cor1['mean'])
              print("task %s Layer %s, gram %s, corr %s"%(str(task),str(evalua),str(gram),str(cor1['mean'])))
            except Exception as e:
              val={}
              val['evaluate']=str(evalua)
              val['gram']=str(gram)
              val['lay']=str(lay)
              val['task']=str(task)
              val['corr']="FailedSVC"
              print("task %s Layer %s, gram %s, corr %s"%(str(task),str(evalua),str(gram),"Failed SVC"))
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
    cnn_array={ 1: []}
    for layer in cnn_encoder_layer1:
      this_wei = layer.weight
      this_wei=this_wei.to('cpu')
      curr_array=this_wei.detach().data.numpy()
      cnn_array[1].append(curr_array)
      print(curr_array.shape)
      this_wei=this_wei.to('cuda')
    for i,layer in enumerate(cnn_encoder_layer2):
      for gram in layer:
        this_wei = gram.weight
        this_wei=this_wei.to('cpu')
        curr_array=this_wei.detach().data.numpy()
        if not ((i+2) in cnn_array):
          cnn_array[(i+2)] = []
        cnn_array[i+2].append(curr_array)
        print(curr_array.shape)
        this_wei=this_wei.to('cuda')
    return cnn_array
