import numpy as np
import matplotlib.pyplot as plt
import pickle
from emnist_helpers import load_emnist

from net import Net
from leNet_layers import Conv, MaxPool
from layers import FC, ReLU
from gradient_descent import SGD
from emnist_helpers import load_emnist



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
        

# Load EMNIST dataset
X_train, Y_train, X_test, Y_test = load_emnist()

# Preprocess data (same as during training)
X_train, X_test = X_train / 255.0, X_test / 255.0
X_train -= np.mean(X_train)
X_test -= np.mean(X_test)
X_train = X_train.reshape(-1, 1, 28, 28).astype(np.float32)  # float32 saves memory
X_test = X_test.reshape(-1, 1, 28, 28).astype(np.float32)

# Initialize model
model = LeNet()

# Load saved weights
with open("lenet_weights_emnist.pkl", "rb") as f:
    saved_weights = pickle.load(f)
model.set_params(saved_weights)

def evaluate_and_plot_misclassified(model, X, Y, batch_size=64):
    num_samples = X.shape[0]
    num_batches = num_samples // batch_size
    correct = 0
    processed = 0
    misclassified_samples = []
    
    print(f"⏳ Evaluating on {num_samples} samples ({num_batches} batches)...")

    for i in range(num_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        X_batch = X[start:end]
        Y_batch = Y[start:end]
        
        Y_pred = model.forward(X_batch)
        predicted = np.argmax(Y_pred, axis=1)
        batch_correct = np.sum(predicted == Y_batch)
        correct += batch_correct
        processed += batch_size

        # Store misclassified images
        misclassified_idx = np.where(predicted != Y_batch)[0]
        for idx in misclassified_idx:
            misclassified_samples.append((X_batch[idx], Y_batch[idx], predicted[idx]))

    # Handle remaining samples
    if num_samples % batch_size != 0:
        remaining = num_samples - (num_batches * batch_size)
        X_batch = X[-remaining:]
        Y_batch = Y[-remaining:]
        
        print(f"  ↳ Processing final batch of {remaining} samples...")
        Y_pred = model.forward(X_batch)
        predicted = np.argmax(Y_pred, axis=1)
        batch_correct = np.sum(predicted == Y_batch)
        correct += batch_correct
        processed += remaining

        # Store misclassified images
        misclassified_idx = np.where(predicted != Y_batch)[0]
        for idx in misclassified_idx:
            misclassified_samples.append((X_batch[idx], Y_batch[idx], predicted[idx]))

    accuracy = correct / num_samples
    print(f"✅ Evaluation complete! Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    
    # Plot misclassified images
    print(f"📌 Found {len(misclassified_samples)} misclassified samples. Plotting them one by one...")

    for img, true_label, pred_label in misclassified_samples:
        plt.imshow(img.squeeze(), cmap='gray')
        plt.title(f"True: {true_label}, Pred: {pred_label}")
        plt.show()
    
    return accuracy

# Evaluate and plot misclassified images
test_acc = evaluate_and_plot_misclassified(model, X_test, Y_test, batch_size=64)
print(f"Test Accuracy: {test_acc*100:.2f}%")

# Evaluate
# train_acc = evaluate(model, X_train, Y_train, batch_size=64)
# test_acc = evaluate(model, X_test, Y_test, batch_size=64)
# print(f"Train Accuracy: {train_acc*100:.2f}%")
# print(f"Test Accuracy: {test_acc*100:.2f}%")

# Test on random samples
# for _ in range(5):
#     idx = np.random.randint(0, len(X_test))
#     x = X_test[idx]
#     y = Y_test[idx]
#     y_pred = np.argmax(model.forward(x[np.newaxis, ...]))  # add batch dim
    
#     plt.imshow(x.squeeze(), cmap='gray')
#     plt.title(f"True: {y}, Pred: {y_pred}")
#     plt.show()