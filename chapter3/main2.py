import nnfs.datasets
import numpy as np
import nnfs
import matplotlib.pyplot as plt

nnfs.init()

class Layer_Dense:
    def __init__ (self, n_inputs, n_neurons):
        #init weights and bises
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        #calculate ouputs
        self.output = np.dot(inputs, self.weights) + self.biases

X, y = nnfs.datasets.spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2, 3)
dense1.forward(X)
print(dense1.output[:5])