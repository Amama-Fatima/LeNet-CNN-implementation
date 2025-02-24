import pickle
import random
import numpy as np
import matplotlib.pyplot as plt

from helpers import MakeOneHot, draw_losses, get_batch, load

from layers import FC, ReLU

from loss import CrossEntropyLoss
from net import Net

from gradient_descent import SGD


class TwoLayerNet(Net):
    def __init__(self, N, D_in, H, D_out, weights=''):
        
        self.FC1 = FC(D_in, H)
        
        self.ReLU1 = ReLU()
        self.FC2 = FC(H, D_out)

        if weights == '':
            print("\nUsing randomly initialized weights")
        else:
            print(f"\nLoading weights from: {weights}")
            with open(weights, 'rb') as f:
                params = pickle.load(f)
                self.set_params(params)
                print("Weights loaded successfully")

    def forward(self, X):
        h1 = self.FC1._forward(X)
        a1 = self.ReLU1._forward(h1)
        h2 = self.FC2._forward(a1)        
        return h2
    
    def backward(self, dout):
        
        dout = self.FC2._backward(dout)
        
        dout = self.ReLU1._backward(dout)
        
        dout = self.FC1._backward(dout)

    def get_params(self):
        print("\nRetrieving network parameters")
        params = [self.FC1.W, self.FC1.b, self.FC2.W, self.FC2.b]
        for i, param in enumerate(params):
            print(f"Parameter {i} shape: {param['val'].shape}")
        return params
    
    def set_params(self, params):
        print("\nSetting network parameters")
        [self.FC1.W, self.FC1.b, self.FC2.W, self.FC2.b] = params



X_train, Y_train, X_test, Y_test = load()
X_train, X_test = X_train/float(255), X_test/float(255) # Normalization. Each pixel will have a value in the range 0 - 255. Diving by 255 help keep the values between 0 and 1
X_train -= np.mean(X_train) # Centering the data. Subtracting the mean of the training data from the training and test data. This centers the data around 0
X_test -= np.mean(X_test)


batch_size = 64
D_in = 784
D_out = 10

### TWO LAYER NET ###
H = 400
model = TwoLayerNet(batch_size, D_in, H, D_out)

losses = []
optim = SGD(model.get_params(), lr=0.0001, reg=0)
criterion = CrossEntropyLoss()

ITER = 25000
for i in range(ITER):
    X_batch, Y_batch = get_batch(X_train, Y_train, batch_size)
    Y_batch = MakeOneHot(Y_batch, D_out)
    Y_pred = model.forward(X_batch)
    loss, dout = criterion.get(Y_pred, Y_batch)
    model.backward(dout)
    optim.step()

    if i % 100 == 0:
        print("%s%% iter: %s, loss: %s" % (100*i/ITER,i, loss))
        losses.append(loss)

# save params
weights = model.get_params()
with open("weights.pkl","wb") as f:
	pickle.dump(weights, f)

draw_losses(losses)


# TRAIN SET ACC
Y_pred = model.forward(X_train)
result = np.argmax(Y_pred, axis=1) - Y_train
result = list(result)
print("TRAIN--> Correct: " + str(result.count(0)) + " out of " + str(X_train.shape[0]) + ", acc=" + str(result.count(0)/X_train.shape[0]))

# TEST SET ACC
Y_pred = model.forward(X_test)
result = np.argmax(Y_pred, axis=1) - Y_test
result = list(result)
print("TEST--> Correct: " + str(result.count(0)) + " out of " + str(X_test.shape[0]))

# test on random samples
for i in range(5):
    idx = random.randint(0, X_test.shape[0])
    x = X_test[idx]
    y = Y_test[idx]
    y_pred = model.forward(x)
    print("True label: %s, Predicted label: %s" % (y, np.argmax(y_pred)))

# display image 
plt.imshow(x.reshape(28, 28), cmap='gray')
plt.show()
