import numpy as np

dvalues = np.array([[1., 1., 1.],
                    [2., 2., 2.],
                    [3., 3., 3.]])

biases = np.array([[2, 3, 0.5]])

dbiases = np.sum(dvalues, axis=0, keepdims=True)
print(dbiases)
