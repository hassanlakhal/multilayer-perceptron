import pandas as pd
import numpy as np

def load_and_split_data(file_path='data.csv', test_size=0.2):
    df = pd.read_csv(file_path, header=None)
    
    X = df.iloc[:, 2:].values.astype(float)
    y = df.iloc[:, 1].values
    
    y = np.where(y == 'M', 1, 0).reshape(-1, 1).astype(float)
    
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X = (X - X_min) / (X_max - X_min + 1e-8) 
    
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    split_at = int(n_samples * (1 - test_size))
    
    train_idx = indices[:split_at]
    test_idx = indices[split_at:]
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    return X_train, X_test, y_train, y_test