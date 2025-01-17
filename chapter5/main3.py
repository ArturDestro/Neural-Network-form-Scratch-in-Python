import nnfs.datasets
import numpy as np
import nnfs
import matplotlib.pyplot as plt
import math

nnfs.init()

class Loss:
    def calculate(self, output, y):
        #apply the specific loss function
        sample_losses = self.forward(output, y)

        #calculate mean
        data_loss = np.mean(sample_losses)

        return data_loss
    

class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)

        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples),y_true]

        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis = 1)

        return -np.log(correct_confidences)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims=True))

        probabilities = exp_values/np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Layer_Dense:
    def __init__ (self, n_inputs, n_neurons):
        #init weights and bises
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        #calculate ouputs
        self.output = np.dot(inputs, self.weights) + self.biases

X, y = nnfs.datasets.spiral_data(samples=100, classes=3)


#create a dense layer 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)
#create a relu activation
activation1 = Activation_ReLU()
#output layer
dense2 = Layer_Dense(3,3)
#create a softmax activation for the output
activation2 = Activation_Softmax()
#create loss funcion
loss_function = Loss_CategoricalCrossEntropy()

#do the maths
dense1.forward(X)
#apply the hidden layer activation fuction
activation1.forward(dense1.output)
#do the maths
dense2.forward(activation1.output)
#apply the output activation funcion
activation2.forward(dense2.output)

print(activation2.output[:5])

#CALCULATE LOSS

loss = loss_function.calculate(activation2.output, y)

print(loss)

#CALCULATE ACCURACY

predictions = np.argmax(activation2.output, axis=1)

if len(y.shape) == 2:
    y = np.argmax(y, axis=1)

accuracy = np.mean(predictions==y)

print('acc: ', accuracy)