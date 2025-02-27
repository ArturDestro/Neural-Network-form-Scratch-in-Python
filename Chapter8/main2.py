import numpy as np 

dvalues = np.array([[1., 1., 1.],
                    [2., 2., 2.],
                    [3., 3., 3.]])


weights = np.array([[.2, .8, -.5 , 1],
                    [.5, -.91, .26, -.5], 
                    [-.26, -.27, .17, .87]]).T


dinputs = np.dot(dvalues, weights.T)

print(dinputs)