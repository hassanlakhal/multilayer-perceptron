import numpy as np
def loss_BCE(y_true, y_pred):
    
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    return loss


def loss_CCE(y_true, y_pred):
   
    y_pred = np.clip(y_pred, 1e-15, 1.0 - 1e-15)
    
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=0))