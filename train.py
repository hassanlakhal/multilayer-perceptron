import src.layers as layers
from src.network import Model
import numpy as np
from split_data import load_and_split_data
from src.utils import save_model, save_history
model = Model()

input_shape = 30
output_shape = 2
optimizer = 'rmsprop'
network = model.createNetwork([
layers.DenseLayer(input_shape, activation='sigmoid',optimizer=optimizer),
layers.DenseLayer(24, activation='sigmoid', weights_initializer='heUniform',optimizer=optimizer),
layers.DenseLayer(24, activation='sigmoid', weights_initializer='heUniform',optimizer=optimizer),
layers.DenseLayer(24, activation='sigmoid', weights_initializer='heUniform',optimizer=optimizer),
layers.DenseLayer(output_shape, activation='softmax', weights_initializer='heUniform',optimizer=optimizer)
])


X_train, X_test, y_train, y_test = load_and_split_data('data/data.csv')


history = model.fit(network, (X_train, y_train), (X_test, y_test), 
                    loss='categoricalCrossentropy', 
                    learning_rate=0.001, 
                    batch_size=8, 
                    epochs=70)

save_model(network)


history_filename = f"model/history_{optimizer}.json"
save_history(history, history_filename)