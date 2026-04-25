import src.layers as layers
from src.network import Model
import numpy as np
from split_data import load_and_split_data
model = Model()


def load_model(network, filename='model/saved_model.npy'):
    
    model_data = np.load(filename, allow_pickle=True)
    
    data_idx = 0
    for layer in network.layers:
        if hasattr(layer, 'weights'):
            layer.weights = model_data[data_idx]
            layer.bias = model_data[data_idx + 1]
            data_idx += 2
            
    print("Model loaded successfully from .npy!")


network = model.createNetwork([
layers.DenseLayer(30, activation='sigmoid'),
layers.DenseLayer(24, activation='sigmoid', weights_initializer='heUniform'),
layers.DenseLayer(24, activation='sigmoid', weights_initializer='heUniform'),
layers.DenseLayer(24, activation='sigmoid', weights_initializer='heUniform'),
layers.DenseLayer(2, activation='softmax', weights_initializer='heUniform')
])



load_model(network)

_, X_test, _, y_test = load_and_split_data('data/data.csv')

output = X_test.T 
for layer in network.layers:
    output = layer.forward(output)


y_pred = output[0]       
y_true = y_test.T[0]     
epsilon = 1e-15
y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

bce_loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

print(f"Final BCE Loss on Test Set: {bce_loss:.4f}")