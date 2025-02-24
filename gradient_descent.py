import numpy as np


class SGD():
    def __init__(self, params, lr=0.001, reg=0):
        
        self.parameters = params
        self.lr = lr  # learning rate
        self.reg = reg  # regularization strength
        
        # # Print initial parameter shapes
        # for i, param in enumerate(self.parameters):
        #     print(f"Parameter {i} shape: {param['val'].shape}")

    def step(self):
        
        for i, param in enumerate(self.parameters):
            
            # Calculate regularization term
            reg_term = self.reg * param['val']
            # Calculate total update
            update = self.lr * param['grad'] + reg_term
            # Perform update
            param['val'] -= update
            # Print statistics after update
