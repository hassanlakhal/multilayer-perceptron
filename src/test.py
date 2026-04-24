import layers
from network import Model
import numpy as np
model = Model()

input_shape = 30
output_shape = 2

network = model.createNetwork([
layers.DenseLayer(input_shape, activation='sigmoid'),
layers.DenseLayer(24, activation='sigmoid', weights_initializer='heUniform'),
layers.DenseLayer(24, activation='sigmoid', weights_initializer='heUniform'),
layers.DenseLayer(24, activation='sigmoid', weights_initializer='heUniform'),
layers.DenseLayer(output_shape, activation='softmax', weights_initializer='heUniform')
])


X_train = np.random.randn(100, 30) 
y_train = np.eye(2)[np.random.randint(0, 2, 100)]


X_val = np.random.randn(20, 30)
y_val = np.eye(2)[np.random.randint(0, 2, 20)]


data_train = (X_train, y_train)
data_valid = (X_val, y_val)


model.fit(network, data_train, data_valid, loss='categoricalCrossentropy', learning_rate=0.0314, batch_size=8, epochs=84)

print(network.layers[-1].weights.shape)