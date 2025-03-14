import nnfs
import nnfs.datasets
import numpy as np
import matplotlib.pyplot as plt
import math

nnfs.init()

class Optimizer_Adam:
    def __init__(self, learning_rate=1e-3, decay=0, epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
            (1. / (1. + self.decay*self.iterations))

    def update_params(self, layer):

        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        #update momentum with current gradients
        layer.weight_momentums = self.beta_1 * \
                                layer.weight_momentums + \
                                (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * \
                            layer.bias_momentums + \
                            (1 - self.beta_1) * layer.dbiases
        #get corrected momentums
        weight_momentums_corrected = layer.weight_momentums/ \
        (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums/ \
        (1 - self.beta_1 ** (self.iterations + 1))

        #update cache with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + \
        (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + \
        (1 - self.beta_2) * layer.dbiases**2

        #get corrected cache
        weight_cache_corrected = layer.weight_cache/ \
        (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache/ \
        (1 - self.beta_2 ** (self.iterations + 1))

        layer.weights += -self.current_learning_rate * \
                            weight_momentums_corrected / \
                            (np.sqrt(weight_cache_corrected) + self.epsilon)
        
        layer.biases += -self.current_learning_rate * \
                        bias_momentums_corrected / \
                        (np.sqrt(bias_cache_corrected) + self.epsilon)


    def post_update_params(self):
        self.iterations+=1

# Dense layer
class Layer_Dense:

# Layer initialization
    def __init__(self, inputs, neurons):
        self.weights = 0.01 * np.random.randn(inputs, neurons)
        self.biases = np.zeros((1, neurons))

    # Forward pass
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs
    
    # Backward bass
    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)


# ReLU activation
class Activation_ReLU:
    # Forward pass
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    # Backward pass
    def backward(self, dvalues):
        # Since we need to modify the original variable,
        # let's make a copy of the values first
        self.dinputs = dvalues.copy()
        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0

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

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

            self.dinputs = -y_true / dvalues
            self.dinputs = self.dinputs/samples

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims=True))

        probabilities = exp_values/np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        for idx, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)

            jacobian_matrix = np.diagflat(single_output) - \
            np.dot(single_output, single_output.T)

            
            self.dinputs[idx] = np.dot(jacobian_matrix,
            single_dvalues)

class Activation_Softmax_Loss_CategoricalCrossentropy:
    def __init__ (self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossEntropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output

        return self.loss.calculate(self.output, y_true)
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis = 1)

        self.dinputs = dvalues.copy()

        self.dinputs[range(samples), y_true] -=1

        self.dinputs = self.dinputs / samples


# Create dataset
X, y = nnfs.datasets.spiral_data(samples=500, classes=3)
# Create model

dense1 = Layer_Dense(2, 64) # first dense layer, 2 inputs
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64, 3) # second dense layer, 3 inputs, 3 outputs
activation2 = Activation_Softmax()
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
optimizer = Optimizer_Adam(learning_rate=0.02, decay= 5e-7)
for epoch in range(10001):
    #foward pass through this layer
    dense1.forward(X)

    #apply ReLU
    activation1.forward(dense1.output)

    #ReLU output to the next layer
    dense2.forward(activation1.output)

    loss = loss_activation.forward(dense2.output, y)

    #calculate accuracy from output of activation2 and targets
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis = 1)
    accuracy = np.mean(predictions==y)

    if not epoch % 100:
        print(f'epoch {epoch}, '+ 
              f'acc: {accuracy:.3f}, ' + 
              f'loss: {loss:.3f}, '
              f'lr: {optimizer.current_learning_rate}')

    #Backward pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()
    

# Plotando as fronteiras de decisão
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

grid = np.c_[xx.ravel(), yy.ravel()]
dense1.forward(grid)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
predictions = np.argmax(dense2.output, axis=1)
predictions = predictions.reshape(xx.shape)

plt.contourf(xx, yy, predictions, alpha=0.8, cmap=plt.cm.Spectral)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral, edgecolors='k')
plt.show()

X_test, y_test = nnfs.datasets.spiral_data(samples=100, classes=3)

dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)

loss = loss_activation.forward(dense2.output, y_test)

predictions = np.argmax(loss_activation.output, axis=1)
if len(y_test.shape) == 2:
    y_test = np.argmax(y_test, axis=1)

accuracy = np.mean(predictions==y_test)

print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')

X_test, y_test = nnfs.datasets.spiral_data(samples=100, classes=3)

dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)

loss = loss_activation.forward(dense2.output, y_test)

predictions = np.argmax(loss_activation.output, axis=1)
if len(y_test.shape) == 2:
    y_test = np.argmax(y_test, axis=1)

accuracy = np.mean(predictions==y_test)

print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')