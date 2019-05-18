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
  return r.mean()


def get_correlation_for_two(cca_first, cca_second):
  vect=[]
  for filt in range(len(cca_first)):
    vect.append(ComputeCCA(cca_first[filt], cca_second[filt]))
  cca_now= sum(vect)/len(vect)
  print("Got CCA", cca_now)
  return cca_now
