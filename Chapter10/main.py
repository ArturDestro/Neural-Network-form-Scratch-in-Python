class Optimizer_SGD:
    def __init__(self, learning_rate=1):
        self.learning_rate = learning_rate

    def update_parameters(self, layer):
        layer.weights+= - self.learning_rate * layer.dweights
        layer.biases +- - self.learning_rate * layer.dbiases