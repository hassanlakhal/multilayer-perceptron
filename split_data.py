import pandas as pd
import numpy as np

def load_and_split_data(file_path='data.csv', test_size=0.2):
    df = pd.read_csv(file_path, header=None)
    
    X = df.iloc[:, 2:].values.astype(float)
    
    labels = df.iloc[:, 1].values
    y = np.zeros((len(labels), 2))
    y[labels == 'M'] = [1, 0] # Malignant
    y[labels == 'B'] = [0, 1] # Benign
    
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0) + 1e-8
    X = (X - X_mean) / X_std
    
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    split_at = int(n_samples * (1 - test_size))
    
    train_idx = indices[:split_at]
    valid_idx = indices[split_at:]
    
    X_train, X_valid = X[train_idx], X[valid_idx]
    y_train, y_valid = y[train_idx], y[valid_idx]
    
    print(f"x_train shape : {X_train.shape}")
    print(f"x_valid shape : {X_valid.shape}")
    
    return X_train, X_valid, y_train, y_valid