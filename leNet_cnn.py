import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import tracemalloc

from net import Net
from leNet_layers import Conv, MaxPool
from layers import FC, ReLU, Softmax
from gradient_descent import SGD
from loss import CrossEntropyLoss
from helpers import MakeOneHot, draw_losses, get_batch, load



class LeNet(Net):
    def __init__(self):
        self.conv1 = Conv(1, 6, 5)
        self.ReLU1 = ReLU()
        self.pool1 = MaxPool(2, 2)
        self.conv2 = Conv(6, 16, 5)
        self.ReLU2 = ReLU()
        self.pool2 = MaxPool(2, 2)
        self.FC1 = FC(16 * 4 * 4, 120)
        self.ReLU3 = ReLU()
        self.FC2 = FC(120, 84)
        self.ReLU4 = ReLU()
        self.FC3 = FC(84, 10)
        self.Softmax = Softmax()

        self.p2_shape = None

    def forward(self, X):
        try:
            h1 = self.conv1._forward(X)
            
            a1 = self.ReLU1._forward(h1)
            
            p1 = self.pool1._forward(a1)
            
            h2 = self.conv2._forward(p1)
            
            a2 = self.ReLU2._forward(h2)
            
            p2 = self.pool2._forward(a2)
            
            self.p2_shape = p2.shape
            fl = p2.reshape(X.shape[0], -1)
            
            h3 = self.FC1._forward(fl)
            a3 = self.ReLU3._forward(h3)
            h4 = self.FC2._forward(a3)
            a4 = self.ReLU4._forward(h4)
            h5 = self.FC3._forward(a4)
            return h5
        except Exception as e:
            print(f"Error in forward pass: {e}")
            print(f"Input tensor shape: {X.shape}")
            raise 
    
    def backward(self, dout):
        try:
            if np.isnan(dout).any():
                print(f"NaN detected in dout at last layer")
                exit()
            dout = self.FC3._backward(dout)
            if np.isnan(dout).any():
                print(f"NaN detected in dout after fc3 backward")
                exit()
            
            dout = self.ReLU4._backward(dout)
            if np.isnan(dout).any():
                print(f"NaN detected in dout after ReLU4 backward")
                exit()
            
            dout = self.FC2._backward(dout)
            if np.isnan(dout).any():
                print(f"NaN detected in dout after FC2 backward")
                exit()
            
            dout = self.ReLU3._backward(dout)
            if np.isnan(dout).any():
                print(f"NaN detected in dout after ReLU3 backward")
                exit()
            
            dout = self.FC1._backward(dout)
            if np.isnan(dout).any():
                print(f"NaN detected in dout after FC1 backward")
                exit()
            
            # Reshape for conv layers
            dout = dout.reshape(self.p2_shape)
            if np.isnan(dout).any():
                print(f"NaN detected in dout after reshaping")
                exit()
            
            dout = self.pool2._backward(dout)
            if np.isnan(dout).any():
                print(f"NaN detected in dout after pool2 backward")
                exit()
            
            dout = self.ReLU2._backward(dout)
            if np.isnan(dout).any():
                print(f"NaN detected in dout after ReLU2 backward")
                exit()
            
            dout = self.conv2._backward(dout)
            if np.isnan(dout).any():
                print(f"NaN detected in dout after conv2 backward")
                exit()
            
            dout = self.pool1._backward(dout)
            if np.isnan(dout).any():
                print(f"NaN detected in dout after pool1 backward")
                exit()
            
            dout = self.ReLU1._backward(dout)
            if np.isnan(dout).any():
                print(f"NaN detected in dout after ReLU1 backward")
                exit()
            
            dout = self.conv1._backward(dout)
            if np.isnan(dout).any():
                print(f"NaN detected in dout after conv1 backward")
                exit()
            
        except Exception as e:
            print(f"Error in backward pass: {e}")
            print(f"Current dout shape: {dout.shape}")
            raise

    def get_params(self):
        return [self.conv1.W, self.conv1.b, self.conv2.W, self.conv2.b, self.FC1.W, self.FC1.b, self.FC2.W, self.FC2.b, self.FC3.W, self.FC3.b]

    def set_params(self, params):
        [self.conv1.W, self.conv1.b, self.conv2.W, self.conv2.b, self.FC1.W, self.FC1.b, self.FC2.W, self.FC2.b, self.FC3.W, self.FC3.b] = params
        



# Load dataset
X_train, Y_train, X_test, Y_test = load()

# Normalize and center the data
X_train, X_test = X_train / 255.0, X_test / 255.0
X_train -= np.mean(X_train)
X_test -= np.mean(X_test)

X_train = X_train.reshape(-1, 1, 28, 28)  # (batch_size, channels, height, width)
X_test = X_test.reshape(-1, 1, 28, 28)

batch_size = 64
D_out = 10  # Number of classes

# Initialize LeNet model
model = LeNet()

# Loss function and optimizer
losses = []
optim = SGD(model.get_params(), lr=0.001, momentum=0.9, reg=1e-4)
criterion = CrossEntropyLoss()

# Training loop
ITER = 25000
tracemalloc.start()
np.seterr(over='ignore')  # Prevent overflow warnings
np.set_printoptions(precision=4, suppress=True)
for i in range(ITER):
    X_batch, Y_batch = get_batch(X_train, Y_train, batch_size)
    Y_batch = MakeOneHot(Y_batch, D_out)
    Y_pred = model.forward(X_batch)
    loss, dout = criterion.get(Y_pred, Y_batch)
    model.backward(dout)
    optim.step()

    if i % 10 == 0:
        print(f"{100 * i / ITER:.2f}% iter: {i}, loss: {loss}")
        current, peak = tracemalloc.get_traced_memory()
        print(f"Memory: {current/1e6:.2f}MB (Peak: {peak/1e6:.2f}MB)")
        losses.append(loss)

# Save trained model weights
weights = model.get_params()
with open("lenet_weights.pkl", "wb") as f:
    pickle.dump(weights, f)

draw_losses(losses)

# Evaluate accuracy on training set
Y_pred = model.forward(X_train)
result = np.argmax(Y_pred, axis=1) - Y_train
print(f"TRAIN--> Correct: {list(result).count(0)} out of {X_train.shape[0]}, acc={list(result).count(0) / X_train.shape[0]:.4f}")

# Evaluate accuracy on test set
Y_pred = model.forward(X_test)
result = np.argmax(Y_pred, axis=1) - Y_test
print(f"TEST--> Correct: {list(result).count(0)} out of {X_test.shape[0]}")

# Test on random samples
for i in range(5):
    idx = random.randint(0, X_test.shape[0] - 1)
    x = X_test[idx]
    y = Y_test[idx]
    y_pred = model.forward(x[np.newaxis, ...])  # Add batch dimension
    print(f"True label: {y}, Predicted label: {np.argmax(y_pred)}")

    # Display image
    plt.imshow(x.squeeze(), cmap='gray')
    plt.show()