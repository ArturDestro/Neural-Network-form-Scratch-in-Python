#3 NEURONS
import numpy as np

inputList = [[1, 2, 3, 2.5],
             [2, 5, -1, 2],
             [-1.5, 2.7, 3.3, -0.8]]
#matrix of weights correponded to each neuron input
weightsList = [[0.2, 0.8, -0.5, 1],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87]]

#bias of each neuron
biasList = [2, 3, 0.5]

#outputlayer
layerOutput = []

"""
#loops that does the maths
for weights, bias in zip(weightsList, biasList):

    neuronOutput = 0
    
    for input, weight in zip(inputList, weights):
        neuronOutput+=input*weight
    
    neuronOutput+=bias

    layerOutput.append(neuronOutput)
"""

layerOutput = np.dot(inputList, np.array(weightsList).T) + biasList

print(layerOutput)

