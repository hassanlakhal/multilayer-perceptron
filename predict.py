import argparse
import numpy as np
import src.layers as layers
from src.network import Model
from src.utils import load_model
from split_data import load_and_split_data

def evaluate():
    parser = argparse.ArgumentParser(description="Breast Cancer Model Evaluation Tool")
    
    parser.add_argument('--data', type=str, default='data/data.csv', 
                        help='Path to the dataset CSV file')
    parser.add_argument('--model_path', type=str, default='model/best_model.npy', 
                        help='Path to the saved model weights (.npy file)')
    
    parser.add_argument('--layers', type=int, nargs='+', default=[24, 24, 24], 
                        help='Space-separated list of hidden layer sizes (e.g., --layers 24 24 24)')
    
    parser.add_argument('--activation', type=str, default='sigmoid', choices=['sigmoid'],
                        help='Activation function used in hidden layers (default: sigmoid)')

    args = parser.parse_args()

    model = Model()
    structure = [layers.DenseLayer(30, activation=args.activation)]
    
    for neurons in args.layers:
        structure.append(layers.DenseLayer(neurons, activation=args.activation, weights_initializer='heUniform'))
    
    structure.append(layers.DenseLayer(2, activation='softmax', weights_initializer='heUniform'))
    
    network = model.createNetwork(structure)

    try:
        load_model(network, args.model_path)
        _, X_test, _, y_test = load_and_split_data(args.data)
        print(f"\n[INFO] Model loaded successfully.")
        print(f"[INFO] Architecture: 30 -> {' -> '.join(map(str, args.layers))} -> 2")
    except Exception as e:
        print(f"[ERROR] Failed to load model or data: {e}")
        return

    output = X_test.T 
    for layer in network.layers:
        output = layer.forward(output)

    y_pred = output[0]       
    y_true = y_test.T[0]     
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    y_pred_binary = (y_pred >= 0.5).astype(int)
    y_true = y_true.flatten()
    y_pred_binary = y_pred_binary.flatten()

    TP = np.sum((y_true == 1) & (y_pred_binary == 1))
    TN = np.sum((y_true == 0) & (y_pred_binary == 0))
    FP = np.sum((y_true == 0) & (y_pred_binary == 1))
    FN = np.sum((y_true == 1) & (y_pred_binary == 0))

    accuracy = (TP + TN) / len(y_true)
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

    print(f"\n" + "="*30)
    print(f"   EVALUATION RESULTS")
    print(f"="*30)
    print(f"Accuracy:  {accuracy*100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1_score:.4f}")
    print(f"-"*30)
    print("Confusion Matrix:")
    print(f"              Predicted 0   Predicted 1")
    print(f"Actual 0 (B):     {TN:<10}    {FP:<10}")
    print(f"Actual 1 (M):     {FN:<10}    {TP:<10}")
    print(f"="*30 + "\n")

if __name__ == "__main__":
    evaluate()