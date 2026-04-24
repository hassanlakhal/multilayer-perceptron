import numpy as np

class Model:
    def createNetwork(self, layers_list):
        class NetworkContainer:
            def __init__(self, layers):
                self.layers = layers
        
        
        for i in range(1, len(layers_list)):
            input_dim = layers_list[i-1].units
            print(f"input_dim {input_dim}")
            layers_list[i].initialize(input_dim)
            
        return NetworkContainer(layers_list)
    
    def fit(self, network, data_train, data_valid, loss='categoricalCrossentropy', learning_rate=0.0314,
                batch_size=8, epochs=84)
        
        X_train, y_train = data_train
        X_val, y_val = data_valid
        n_samples = X_train.shape[0]

        for i in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffle = X_train[indices]
            y_suffle = y_train[indices]
