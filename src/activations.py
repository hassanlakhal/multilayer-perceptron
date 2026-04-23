import numpy as np
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)