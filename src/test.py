import layers
from network import Model

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

print(network.layers[-1].weights.shape)