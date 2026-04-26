import numpy as np
from src.activations import sigmoid , softmax
class DenseLayer :
    def __init__(self, units, activation='sigmoid', weights_initializer='default'):
        self.units = units
        self.activation = activation
        self.initializer = weights_initializer
        self.weights = None
        self.bias = None

        self.a = None
        self.input = None
        self.z = None
        self.dW = None
        self.dB = None

        self.v_w = np.zeros_like(self.weights) 
        self.v_b = np.zeros_like(self.bias)   
    
    def initialize(self, input_size):
        if  self.initializer == 'heUniform':
            limit = np.sqrt(6/ input_size)
            self.weights = np.random.uniform(-limit, limit, (self.units, input_size))
        else:
            self.weights = np.random.randn(self.units, input_size) * 0.01
        
        self.bias = np.zeros((self.units, 1)) 

    def forward(self, input_data):
        self.input = input_data

        self.Z = np.dot(self.weights, input_data) + self.bias
        
        if self.activation == 'sigmoid':
            self.A = sigmoid(self.Z)
        elif self.activation == 'softmax':
            self.A = softmax(self.Z)
        else:
            self.A = self.Z
        
        return self.A

    def backward(self, gradient, lr, beta=0.9, epsilon=1e-8):
        
        self.dW = np.dot(gradient, self.input.T)

        self.dB = np.sum(gradient, axis=1, keepdims=True)


        self.v_w = beta * self.v_w + (1 - beta) * (self.dW**2)
        self.v_b = beta * self.v_b + (1 - beta) * (self.dB**2)
        
        
        self.weights -= (lr / (np.sqrt(self.v_w) + epsilon)) * self.dW
        self.bias -= (lr / (np.sqrt(self.v_b) + epsilon)) * self.dB

        d_input = np.dot(self.weights.T, gradient)


        # self.weights -= lr * self.dW
        # self.bias -= lr *  self.dB

        return d_input
