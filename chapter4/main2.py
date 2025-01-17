import nnfs.datasets
import numpy as np
import nnfs
import matplotlib.pyplot as plt
import math

layer_outputs = [4.8, 1.21, 2.385]

exp_values = np.exp(layer_outputs)

#NORMALIZE VALUES

norm_values = exp_values/np.sum(exp_values)

print('Normalized exponentiated values: ')
print(norm_values)

print("Sum of normalized values ", sum(norm_values))