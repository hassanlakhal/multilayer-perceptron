import numpy as np
from src.loss import loss_BCE
class Model:
    def createNetwork(self, layers_list):
        class NetworkContainer:
            def __init__(self, layers):
                self.layers = layers
                
        
        layers_list[0].initialize(layers_list[0].units)
        for i in range(1, len(layers_list)):
            input_dim = layers_list[i-1].units
            print(f"input_dim {input_dim}")
            layers_list[i].initialize(input_dim)
            
        return NetworkContainer(layers_list)
    
    def fit(self, network, data_train, data_valid, loss='categoricalCrossentropy', learning_rate=0.0314,
                batch_size=8, epochs=84):
        
        X_train, y_train = data_train
        n_samples = X_train.shape[0]

        for epoch in range(epochs): 
            current_epoch_loss = 0.0 
            num_batches = 0

            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            for b in range(0, n_samples, batch_size): 
                X_batch = X_shuffled[b:b+batch_size]
                y_batch = y_shuffled[b:b+batch_size]

                output = X_batch
                for layer in network.layers:
                    output = layer.forward(output)
              
                batch_loss = loss_BCE(y_batch, output, loss)
                current_epoch_loss += batch_loss
                num_batches += 1

            print(f"Epoch {epoch+1}/{epochs} - Loss: {current_epoch_loss / num_batches}")
