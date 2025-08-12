
# LeNet-CNN-Implementation 🚀

A NumPy-based implementation of the LeNet convolutional neural network 1998 research paper, trained on the EMNIST dataset for handwritten digit recognition.

## 📌 Demo

Interact with the trained model at:
[https://web-production-cce99.up.railway.app/](https://web-production-cce99.up.railway.app/)

https://github.com/user-attachments/assets/4cbab265-41ad-433c-905a-a9ee7c9ea476

---

## 🧠 Model Architecture

This project implements the classic **LeNet-5** architecture using **pure NumPy**, structured as follows:

| Layer Type   | Details                              | Output Shape |
| ------------ | ------------------------------------ | ------------ |
| **Input**    | 1 × 28 × 28 grayscale image          | (1, 28, 28)  |
| **Conv1**    | 6 filters, 5×5 kernel, stride=1      | (6, 24, 24)  |
| **ReLU**     | Activation                           | (6, 24, 24)  |
| **MaxPool1** | 2×2 pooling                          | (6, 12, 12)  |
| **Conv2**    | 16 filters, 5×5 kernel, stride=1     | (16, 8, 8)   |
| **ReLU**     | Activation                           | (16, 8, 8)   |
| **MaxPool2** | 2×2 pooling                          | (16, 4, 4)   |
| **Flatten**  | -                                    | (256,)       |
| **FC1**      | Fully Connected → 120 units          | (120,)       |
| **ReLU**     | Activation                           | (120,)       |
| **FC2**      | Fully Connected → 84 units           | (84,)        |
| **ReLU**     | Activation                           | (84,)        |
| **FC3**      | Fully Connected → 10 units (classes) | (10,)        |
| **Softmax**  | *(applied in loss)*                  | (10,)        |

> 💡 Uses ReLU activations throughout instead of tanh, and MaxPooling instead of average pooling for better gradient flow.

---

## 🔧 Loss Function

This project uses a **combined Softmax + Categorical Cross‑Entropy Loss**, implemented as a class `CrossEntropyLoss`, to train the LeNet model.

### 🔹 Components

1. **Softmax Activation**
2. ***Negative Log-Likelihood Loss (NLLLoss)**
3. **Gradient for Backpropagation**

### 🔹 `CrossEntropyLoss.get` Workflow

* Applies `Softmax._forward()` to transform logits into probabilities.
* Uses `NLLLoss()` to compute the scalar loss.
* Constructs `dout`, the gradient of loss with respect to logits, as:

  ```python
  dout = prob.copy()
  dout[np.arange(N), true_labels] -= 1
  dout /= N
  ```
* Returns both **loss** and **dout**, ready for backpropagation through the network.




