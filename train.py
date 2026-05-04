import src.layers as layers
from src.network import Model
import numpy as np
from split_data import load_and_split_data
from src.utils import save_model, save_history
import matplotlib.pyplot as plt

model = Model()

import argparse

def plot_history(history):
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.plot(history['loss'], label='training loss', color='#1f77b4', linewidth=2)
    ax1.plot(history['val_loss'], label='validation loss', color='#ff7f0e', linestyle='--', linewidth=1.5)
    ax1.set_title('Figure IV.1: Loss')
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()

    ax2.plot(history['accuracy'], label='training acc', color='#1f77b4', linewidth=2)
    ax2.plot(history['val_accuracy'], label='validation acc', color='#ff7f0e', linewidth=2)
    ax2.set_title('Figure IV.2: Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0.4, 1.02) 
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    
    image_name = f"learning_curves_{opt}.png"
    plt.savefig(image_name, dpi=300)
    print(f"Graph saved as {image_name}")


def get_args():
    parser = argparse.ArgumentParser(description="Multilayer Perceptron Training")
    
    parser.add_argument('--layers', nargs='+', type=int, default=[24, 24, 24], help='List of hidden layer units')
    parser.add_argument('--epochs', type=int, default=84, help='Number of epochs')
    parser.add_argument('--loss', type=str, default='categoricalCrossentropy', help='Loss function')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0314, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='sgd', help='Optimizer Algo (sgd or rmsprop)')
    return parser.parse_args()


if __name__ ==  '__main__':

    args = get_args()
    
    opt = args.optimizer
    loss = args.loss
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    epochs = args.epochs

    input_layer = [layers.DenseLayer(30, activation='sigmoid', weights_initializer='heUniform', optimizer=opt)]
    hidden_layers = []
    for units in args.layers :
        hidden_layers.append(layers.DenseLayer(units, activation='sigmoid', weights_initializer='heUniform', optimizer=opt))
        print(units)
    output_layer = [layers.DenseLayer(2, activation='sigmoid', weights_initializer='heUniform', optimizer=opt)]

    all_layers = input_layer + hidden_layers + output_layer
    network = model.createNetwork(all_layers)

    X_train, X_test, y_train, y_test = load_and_split_data('data/data.csv')


    history = model.fit(network, (X_train, y_train), (X_test, y_test), 
                    loss=loss, 
                    learning_rate=learning_rate, 
                    batch_size=batch_size, 
                    epochs=epochs)
                    
    plot_history(history)
    history_filename = f"model/history_{opt}.json"
    save_history(history, history_filename)