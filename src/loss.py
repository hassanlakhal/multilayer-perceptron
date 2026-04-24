import numpy as np
def loss_BCE(y_true, y_pred, type_loss):
    
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = 0
    if type_loss == 'categoricalCrossentropy':
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    return loss