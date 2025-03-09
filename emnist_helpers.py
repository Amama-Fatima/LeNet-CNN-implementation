import numpy as np
import matplotlib.pyplot as plt
import pickle

def save_emnist():
    emnist = {}
    
    # Load training images with rotation/flip correction
    with open("emnist/emnist-digits-train-images-idx3-ubyte", 'rb') as f:
        train_images = np.frombuffer(f.read(), np.uint8, offset=16)
        train_images = train_images.reshape(-1, 28, 28)
        
        # Apply rotation and flip to each image
        corrected_images = np.zeros_like(train_images)
        for i in range(len(train_images)):
            # Rotate 90 degrees clockwise and flip horizontally
            corrected_images[i] = np.fliplr(np.rot90(train_images[i], k=-1))
        
        emnist["training_images"] = corrected_images.reshape(-1, 28*28)
    
    # Load test images with same correction
    with open("emnist/emnist-digits-test-images-idx3-ubyte", 'rb') as f:
        test_images = np.frombuffer(f.read(), np.uint8, offset=16)
        test_images = test_images.reshape(-1, 28, 28)
        
        # Apply same correction to test images
        corrected_test = np.zeros_like(test_images)
        for i in range(len(test_images)):
            corrected_test[i] = np.fliplr(np.rot90(test_images[i], k=-1))
            
        emnist["test_images"] = corrected_test.reshape(-1, 28*28)
    
    # Load labels (same as before)
    with open("emnist/emnist-digits-train-labels-idx1-ubyte", 'rb') as f:
        emnist["training_labels"] = np.frombuffer(f.read(), np.uint8, offset=8)
    
    with open("emnist/emnist-digits-test-labels-idx1-ubyte", 'rb') as f:
        emnist["test_labels"] = np.frombuffer(f.read(), np.uint8, offset=8)
    
    # Save to pickle
    with open("emnist_digits.pkl", 'wb') as f:
        pickle.dump(emnist, f)
    print("EMNIST Digits saved!")


def load_emnist():
    with open("emnist_digits.pkl", 'rb') as f:
        emnist = pickle.load(f)
    return (
        emnist["training_images"], 
        emnist["training_labels"], 
        emnist["test_images"], 
        emnist["test_labels"]
    )
# save_emnist()