import numpy as np

def save_model(network, filename='model/saved_model.npy'):
    model_data = []
    for layer in network.layers:
        if hasattr(layer, 'weights'):
            model_data.append(layer.weights)
            model_data.append(layer.bias)
  
    as_array = np.array(model_data, dtype=object)
    np.save(filename, as_array, allow_pickle=True)
    
    print(f"Model saved to {filename}")


def load_model(network, filename='model/saved_model.npy'):
    
    model_data = np.load(filename, allow_pickle=True)
    
    data_idx = 0
    for layer in network.layers:
        if hasattr(layer, 'weights'):
            layer.weights = model_data[data_idx]
            layer.bias = model_data[data_idx + 1]
            data_idx += 2
            
    print("Model loaded successfully from .npy!")