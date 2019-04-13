import torch
from models.cnn_encoder import CnnEncoder
import pickle

class SaveWeights:

  def __init__(self, encoder_):
    self.encoder_type = encoder_
    self.tryno=1

  def write_weights(self, model, layer, h_dim, code, after_task,tryno):
    self.tryno=tryno
    self.write_weights(model, layer, h_dim, code, after_task)
    
  def write_weights(self, model, layer, h_dim, code, after_task):
    if self.encoder_type == "cnn":
      cnn_array=self.get_cnn_weights(model)
      cnn_array['layer'] = layer
      cnn_array['h_dim'] = h_dim
      cnn_array['code'] = code
      cnn_array['after_task'] = after_task
      cnn_array['tryno'] = tryno
      pickle_out = open("weights/cnn_layer_%s_hdim_%s_CODE_%s_tryno_%s_AFTER_%s.weights"%(str(layer),str(h_dim),str(code),str(self.tryno),str(after_task)), "wb")
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
