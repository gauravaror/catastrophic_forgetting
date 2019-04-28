# This file calculates the CCA between the layers
import pickle

import sys
import argparse
import scipy
from scipy import linalg
from scipy.linalg import decomp_qr
from sklearn import preprocessing
import numpy as np
import pandas as pd

sys.path.append("..")
from svcca import cca_core as svc

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('firstfile', type=str,
                    help='First file for correlation')
parser.add_argument('secondfile', type=str,
                    help='Second file for correlation')
parser.add_argument('thirdfile', type=str,
                    help='Second file for correlation')
parser.add_argument('fourthfile', type=str,
                    help='Second file for correlation')
parser.add_argument('--debug', action='store_true',
                    help='Print the debug messages')

args = parser.parse_args()

# Taken from https://github.com/ytsvetko/qvec/blob/master/qvec_cca.py
def NormCenterMatrix(M):
  M = preprocessing.normalize(M)
  m_mean = M.mean(axis=0)
  M -= m_mean
  return M

def ComputeCCA(X, Y):
  assert X.shape[0] == Y.shape[0], (X.shape, Y.shape, "Unequal number of rows")
  assert X.shape[0] > 1, (X.shape, "Must have more than 1 row")
  
  X = NormCenterMatrix(X)
  Y = NormCenterMatrix(Y)
  X_q, _, _ = decomp_qr.qr(X, overwrite_a=True, mode='economic', pivoting=True)
  Y_q, _, _ = decomp_qr.qr(Y, overwrite_a=True, mode='economic', pivoting=True)
  C = np.dot(X_q.T, Y_q)
  r = linalg.svd(C, full_matrices=False, compute_uv=False)
  d = min(X.shape[1], Y.shape[1])
  r = r[:d]
  r = np.minimum(np.maximum(r, 0.0), 1.0)  # remove roundoff errs
  if args.debug:
    print(r)
  return r.mean()


print("Finding correlation between: ", args.firstfile, args.secondfile, args.thirdfile, args.fourthfile)

first=pickle.load(open(args.firstfile, "rb"))
second=pickle.load(open(args.secondfile, "rb"))
third=pickle.load(open(args.thirdfile, "rb"))
fourth=pickle.load(open(args.fourthfile, "rb"))

def get_correlation_for_two(cca_first, cca_second):
  data=[]
  for layer in range(1,6):
    if (layer in cca_first) and (layer in cca_second):
      for gram in range(3):
        vect=[]
        data_dict={}
        for filt in range(len(first[layer][gram])):
          vect.append(ComputeCCA(cca_first[layer][gram][filt], cca_second[layer][gram][filt]))
          print(cca_second[layer][gram][filt].shape)
          #print("Layer",layer,"gram",gram,"Filt", filt,svc.get_cca_similarity(cca_first[layer][gram][filt], cca_second[layer][gram][filt]))
        #print("Layer %s gram %s : %s"% (str(layer), str(gram), str(sum(vect)/len(vect))))
        data_dict['layer'] = layer
        data_dict['gram'] = gram
        data_dict['correlation'] = sum(vect)/len(vect)
        data_dict['code'] = cca_first['code']
        data_dict['code'] = cca_first['code']
        data.append(data_dict)
  return data

data1=get_correlation_for_two(first,second)
data2=get_correlation_for_two(first,third)
data3=get_correlation_for_two(first,fourth)
final_data=[]
for i in range(len(data1)):
  final_data.append({ 'layer': first['layer'], 
  'h_dim' : first['h_dim'],
  'code' :first['code'],
  'first_task' : first['after_task'],
  'data_layer' : data1[i]['layer'],
  'gram' : data1[i]['gram'],
  'z_first' : data1[i]['correlation'],
  'z_second':data2[i]['correlation'],
  'z_third': data3[i]['correlation']})

df=pd.DataFrame(final_data)
print(df.to_string())

