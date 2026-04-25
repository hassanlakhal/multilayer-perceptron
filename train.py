import src.layers as layers
from src.network import Model
import numpy as np
from split_data import load_and_split_data
model = Model()

def save_model(network, filename='model/saved_model.npy'):
    model_data = []
    for layer in network.layers:
        if hasattr(layer, 'weights'):
            model_data.append(layer.weights)
            model_data.append(layer.bias)
  
    as_array = np.array(model_data, dtype=object)
    np.save(filename, as_array, allow_pickle=True)
    
    print(f"Model saved to {filename}")


input_shape = 30
output_shape = 2

network = model.createNetwork([
layers.DenseLayer(input_shape, activation='sigmoid'),
layers.DenseLayer(24, activation='sigmoid', weights_initializer='heUniform'),
layers.DenseLayer(24, activation='sigmoid', weights_initializer='heUniform'),
layers.DenseLayer(24, activation='sigmoid', weights_initializer='heUniform'),
layers.DenseLayer(output_shape, activation='softmax', weights_initializer='heUniform')
])


X_train, X_test, y_train, y_test = load_and_split_data('data/data.csv')


model.fit(network, (X_train, y_train), (X_test, y_test), loss='categoricalCrossentropy', learning_rate=0.0314, batch_size=8, epochs=70)

save_model(network)