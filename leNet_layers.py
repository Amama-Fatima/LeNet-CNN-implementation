import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod
import gzip
from urllib import request
from helpers import clip_gradients


class Conv():
    """
    Conv layer
    """
    def __init__(self, Cin, Cout, F, stride=1, padding=0, bias=True):
        self.Cin = Cin
        self.Cout = Cout
        self.F = F
        self.S = stride
        #self.W = {'val': np.random.randn(Cout, Cin, F, F), 'grad': 0}
        self.W = {'val': np.random.normal(0.0,np.sqrt(2/Cin),(Cout,Cin,F,F)), 'grad': 0} # Xavier Initialization
        self.b = {'val': np.random.randn(Cout), 'grad': 0}
        self.cache = None
        self.pad = padding

    def _forward(self, X):
        X = np.pad(X, ((0,0),(0,0),(self.pad,self.pad),(self.pad,self.pad)), 'constant')
        N, Cin, H, W = X.shape
        Hout = (H - self.F) // self.S + 1
        Wout = (W - self.F) // self.S + 1
        
        # im2col transformation
        cols = np.zeros((N, Cin, self.F, self.F, Hout, Wout))
        for h in range(self.F):
            h_end = h + Hout*self.S
            for w in range(self.F):
                w_end = w + Wout*self.S
                cols[:, :, h, w, :, :] = X[:, :, h:h_end:self.S, w:w_end:self.S]
        
        cols = cols.transpose(1, 2, 3, 0, 4, 5).reshape(Cin*self.F*self.F, N*Hout*Wout)
        res = self.W['val'].reshape(self.Cout, -1) @ cols
        res = res.reshape(self.Cout, N, Hout, Wout).transpose(1, 0, 2, 3)
        res += self.b['val'].reshape(1, -1, 1, 1)
        
        self.cache = (X, cols)
        return res
    
    def _backward(self, dout):
        X, cols = self.cache
        N, Cin, H, W = X.shape
        # Calculate output dimensions from forward pass
        Hout = (H - self.F) // self.S + 1
        Wout = (W - self.F) // self.S + 1

        dout_flat = dout.transpose(1, 0, 2, 3).reshape(self.Cout, -1)
        
        # Vectorized gradients
        dW = dout_flat @ cols.T
        dW = dW.reshape(self.W['val'].shape)
        db = dout.sum(axis=(0, 2, 3))
        
        W_flat = self.W['val'].reshape(self.Cout, -1)
        dcols = W_flat.T @ dout_flat
        dX = np.zeros_like(X)
        
        # Reverse im2col
        dcols = dcols.reshape(Cin, self.F, self.F, N, Hout, Wout).transpose(3, 0, 1, 2, 4, 5)
        
        for h in range(self.F):
            for w in range(self.F):
                # Adjust slicing to account for stride and avoid over-indexing
                h_start = h
                w_start = w
                h_end = h_start + Hout * self.S
                w_end = w_start + Wout * self.S
                dX[:, :, h_start:h_end:self.S, w_start:w_end:self.S] += dcols[:, :, h, w, :, :]

        self.W['grad'] = dW
        self.b['grad'] = db
        if self.pad > 0:
            dX = dX[:, :, self.pad:-self.pad, self.pad:-self.pad]  # Remove padding
        return dX


class MaxPool():
    def __init__(self, F, stride):
        self.F = F
        self.S = stride
        self.cache = None

    def _forward(self, X):
        N, Cin, H, W = X.shape
        Hout = (H - self.F) // self.S + 1
        Wout = (W - self.F) // self.S + 1

        # Reshape to extract windows
        X_reshaped = X.reshape(N, Cin, Hout, self.F, Wout, self.F)
        X_reshaped = X_reshaped.transpose(0, 1, 2, 4, 3, 5)  # N, Cin, Hout, Wout, F, F
        X_windows = X_reshaped.reshape(-1, self.F * self.F)

        max_vals = X_windows.max(axis=1)
        self.mask = (X_windows == max_vals[:, None])

        counts = self.mask.sum(axis=1)[:, None]
        self.mask = self.mask / counts

        self.mask = self.mask.reshape(N, Cin, Hout, Wout, self.F, self.F)
        return max_vals.reshape(N, Cin, Hout, Wout)

    def _backward(self, dout):
        N, Cin, Hout, Wout = dout.shape
        dout_expanded = dout.reshape(N, Cin, Hout, Wout, 1, 1)
        dX = dout_expanded * self.mask  # Broadcasts to (N, Cin, Hout, Wout, F, F)
        dX = dX.reshape(N, Cin, Hout * self.F, Wout * self.F)
        return dX