import src.layers as layers
from src.network import Model
import numpy as np
from src.utils import load_model
from split_data import load_and_split_data
model = Model()

network = model.createNetwork([
layers.DenseLayer(30, activation='sigmoid'),
layers.DenseLayer(24, activation='sigmoid', weights_initializer='heUniform'),
layers.DenseLayer(24, activation='sigmoid', weights_initializer='heUniform'),
layers.DenseLayer(24, activation='sigmoid', weights_initializer='heUniform'),
layers.DenseLayer(2, activation='softmax', weights_initializer='heUniform')
])


load_model(network, "model/best_model.npy")

_, X_test, _, y_test = load_and_split_data('data/data.csv')

output = X_test.T 
for layer in network.layers:
    output = layer.forward(output)


y_pred = output[0]       
y_true = y_test.T[0]     
epsilon = 1e-15
y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

bce_loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


probabilities = y_pred 

y_pred_binary = (probabilities >= 0.5).astype(int)

y_true = y_true.flatten()
y_pred_binary = y_pred_binary.flatten()

TP = np.sum((y_true == 1) & (y_pred_binary == 1))
TN = np.sum((y_true == 0) & (y_pred_binary == 0))
FP = np.sum((y_true == 0) & (y_pred_binary == 1))
FN = np.sum((y_true == 1) & (y_pred_binary == 0))

precision = TP / (TP + FP + 1e-8)
recall = TP / (TP + FN + 1e-8)
f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

print(f"--- Evaluation Metrics ---\n")
print(f"Final BCE Loss: {bce_loss:.4f}")
print(f"Precision:      {precision:.4f}")
print(f"Recall:         {recall:.4f}")
print(f"F1-Score:       {f1_score:.4f}\n")
print(f"--------------------------")
print("\nConfusion Matrix:")
print(f"               Predicted 0    Predicted 1")
print(f"Actual 0 (B):      {TN}             {FP}")
print(f"Actual 1 (M):      {FN}             {TP}\n")

accuracy = (TP + TN) / len(y_true)
print(f"--------------------------\n")
print(f"Accuracy:       {accuracy*100:.2f}%")
