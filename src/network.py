import numpy as np
from src.loss import loss_BCE, loss_CCE
from src.utils import save_model

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
        X_valid, y_valid = data_valid
        n_samples = X_train.shape[0]
        best_val_loss = float('inf') 
        patience = 10
        patience_counter = 0

        history = {'loss': [], 'val_loss': []}

        for epoch in range(epochs): 
            current_epoch_loss = 0.0 
            num_batches = 0

            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            for b in range(0, n_samples, batch_size): 
                X_batch = X_shuffled[b:b+batch_size].T # (30, batch_size)
                y_batch = y_shuffled[b:b+batch_size].T # (2, batch_size)

                # 1. Forward Pass
                output = X_batch
                for layer in network.layers:
                    output = layer.forward(output)
              
                batch_loss = loss_CCE(y_batch, output)
                current_epoch_loss += batch_loss
                num_batches += 1

          
                error_gradient = output - y_batch 

                gradient = error_gradient
                for layer in reversed(network.layers):
                    gradient = layer.backward(gradient, learning_rate)

            val_output = X_valid.T
            for layer in network.layers:
                val_output = layer.forward(val_output)
            
            epoch_loss = current_epoch_loss / num_batches
            val_loss = loss_BCE(y_valid.T, val_output)
            
            history['loss'].append(epoch_loss)
            history['val_loss'].append(val_loss)

            train_preds = np.argmax(output, axis=0)
            train_true = np.argmax(y_batch, axis=0)
            accuracy = np.mean(train_preds == train_true)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                save_model(network, "model/best_model.npy")
                print(f"--- Model saved at epoch {epoch+1} (Best Val Loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early Stopping! No improvement for {patience} epochs.")
                break

            # print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f} - val_loss: {val_loss:.4f}")
        
        return history
            
