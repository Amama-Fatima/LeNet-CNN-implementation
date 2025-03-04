import random
import numpy as np
import matplotlib.pyplot as plt
import pickle
import gzip

def clip_gradients(grad, threshold=5.0):
    norm = np.linalg.norm(grad)  # Compute L2 norm
    if norm > threshold:
        grad *= (threshold / norm)  # Scale gradients down instead of hard clipping
    return grad


def MakeOneHot(Y, D_out):
    N = Y.shape[0]
    Z = np.zeros((N, D_out))
    Z[np.array(range(N)), Y] = 1
    return Z


def draw_losses(losses):
    t = np.arange(len(losses))
    plt.plot(t, losses)
    plt.show()    

def get_batch(X, Y, batch_size):
    N = len(X)
    i = random.randint(0, N-batch_size)
    return X[i:i+batch_size], Y[i:i+batch_size]

def load():
    with open("mnist.pkl",'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]  