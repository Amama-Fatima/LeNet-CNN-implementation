import numpy as np


class SGD():
    def __init__(self, params, lr=0.001, momentum=0.9, reg=0):
        self.params = params  # Store parameters as instance variable
        self.l = lr
        self.mu = momentum
        self.reg = reg
        self.velocity = [np.zeros_like(p['val']) for p in self.params]  # Use self.params

    def step(self):
        for i, p in enumerate(self.params):  # Reference self.params here
            # Update velocity with regularization
            self.velocity[i] = self.mu * self.velocity[i] + (p['grad'] + self.reg * p['val'])
            # Update parameters
            p['val'] -= self.l * self.velocity[i]