import matplotlib.pyplot as plt
import json

with open('../model/history_sgb.json', 'r') as f: h_sgd = json.load(f)
with open('../model/history_rmsprop.json', 'r') as f: h_rms = json.load(f)

plt.figure(figsize=(10, 6))

plt.plot(h_sgd['val_loss'], label='SGD (Baseline)', color='red', linestyle='--')
plt.plot(h_rms['val_loss'], label='RMSprop (Adaptive)', color='blue', linewidth=2)

plt.title('SGD vs RMSprop: Convergence Comparison')
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()