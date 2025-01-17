import numpy as np

softmax_outputs = np.array([[.7,.2,.1],[.5,.1,.4],[.02,.9,.08]])

class_targets = np.array([0,1,1])

predictions = np.argmax(softmax_outputs, axis=1)

if len(class_targets.shape) == 2:
    class_targets = np.argmax(class_targets, axis=1)

accuracy = np.mean(predictions==class_targets)

print('acc', accuracy)