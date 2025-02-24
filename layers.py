import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod
import gzip
from urllib import request

# Fully connected layer
class FC():
    def __init__(self, D_in, D_out):
        self.cache = None
        self.W = {'val': np.random.normal(0.0, np.sqrt(2/D_in), (D_in, D_out)), 'grad': 0}
        self.b = {'val': np.random.randn(D_out), 'grad': 0}

    def _forward(self, X):
        out = np.dot(X, self.W['val'] + self.b['val'])
        self.cache = X
        return out

    def _backward(self, dout):
        X = self.cache
        dX = np.dot(dout, self.W['val'].T).reshape(X.shape)
        X_reshaped = X.reshape(X.shape[0], np.prod(X.shape[1:]))
        self.W['grad'] = np.dot(X_reshaped.T, dout)
        self.b['grad'] = np.sum(dout, axis=0)
        return dX



class ReLU():
    def __init__(self):
        self.cache = None

    def _forward(self, X):
        out = np.maximum(X, 0)
        self.cache = X
        return out

    def _backward(self, dout):
        X = self.cache
        dX = np.array(dout, copy=True)
        dX[X <= 0] = 0
        return dX


class Softmax():
    def __init__(self):
        self.cache = None
    
    def _forward(self, X):
        maxes = np.amax(X, axis=1)
        maxes = maxes.reshape(maxes.shape[0], 1)
        Y = np.exp(X - maxes)
        row_sums = np.sum(Y, axis=1).reshape(Y.shape[0], 1)
        Z = Y / row_sums
        self.cache = (X, Y, Z)
        return Z
    
    def _backward(self, dout):
        X, Y, Z = self.cache
        N, C = X.shape
        dX = np.zeros_like(X)

        for n in range(N):
            dZ = np.diag(Z[n]) - np.outer(Z[n], Z[n])
            dX[n] = np.dot(dout[n], dZ)
        return dX

