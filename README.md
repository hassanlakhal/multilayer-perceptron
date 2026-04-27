# Multilayer Perceptron

A from-scratch implementation of a Multilayer Perceptron (MLP) Artificial Neural Network in Python using pure NumPy. This project is built to classify binary outcomes, specifically targeted toward identifying Malignant ('M') versus Benign ('B') tumors from a 30-feature dataset (similar to the Breast Cancer Wisconsin dataset).

The project features a custom neural network library created entirely from scratch, bypassing high-level machine learning abstractions like TensorFlow or PyTorch. It heavily emphasizes the mathematical foundations of deep learning.

## Training Performance

![Training Graph](graph.png)

---

## Technical & Theoretical Details
This section explains the foundational theory operating behind the custom implementation found within the `src/` directory.

### 1. Feed-forward Architecture
A Multilayer Perceptron is a fully connected class of feedforward artificial neural networks.
In each `Dense` layer (`src/layers.py`), the inputs vectors undergo a linear transformation followed by a non-linear activation:

$$ Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]} $$
$$ A^{[l]} = g(Z^{[l]}) $$

Where $W^{[l]}$ are the weight matrices, $b^{[l]}$ are biases, and $g$ is the activation function. 
* **Activations (`src/activations.py`)**: 
  - **Sigmoid**: Used for hidden layers, mapping values between 0 and 1. Implementation is highly stabilized via clipping `[-500, 500]` to avoid overflow issues.
  - **Softmax**: Used at the output layer to model class probabilities. Designed safely by subtracting maximum array values for numerical stability (preventing `NaN` scaling).

### 2. Backpropagation
During backpropagation, we compute the error in the final layer and propagate it backward via the chain rule to update the weights.
The error gradient (`error_gradient = output - y_batch`) passes backward:
- Weight gradients: $dW = \frac{1}{m} dZ \cdot A^{[l-1]T}$ 
- Bias gradients: $dB = \frac{1}{m} \sum dZ$

### 3. Optimization Algorithms
Weights are adjusted iteratively to minimize loss. This repository supports two types of gradient descent optimizations:
* **Standard SGD (Stochastic Gradient Descent)**: 
  Updates weights proportionally to the learning rate $\alpha$ and the negative gradient.
  $W = W - \alpha \cdot dW$
* **RMSprop (Root Mean Square Propagation)**:
  An adaptive learning method implemented to speed up convergence by overcoming standard vanishing/exploding gradients. It maintains a decaying average of squared gradients:
  $v_{dw} = \beta v_{dw} + (1 - \beta) dW^2$ 
  $W = W - \frac{\alpha}{\sqrt{v_{dw}} + \epsilon} dW$

### 4. Loss Functions (`src/loss.py`)
Loss functions govern how standard error is quantified during training.
- **Categorical Cross Entropy (CCE)**: Evaluates training predictions natively against one-hot encoded multi-class true labels. 
- **Binary Cross Entropy (BCE)**: Applied strictly on validation checks to test binary correctness. Values are heavily clipped against a minimal threshold ($\epsilon = 1e-15$) to prevent taking logarithms of $0$.

### 5. Training Mechanics (`src/network.py`)
* **Mini-Batch Data Loading**: During epochs, datasets are stochastically shuffled and segmented by `$batch\_size$`, improving memory generalization over full-batch descent.
* **Early Stopping**: Validation loss is tracked dynamically per epoch. If the validation loss plateaus or stops decreasing for $Patience = 10$ consecutive epochs, the model assumes convergence, saves the lowest loss (`best_model.npy`), and safely terminates.
* **Weight Initializations**: He Uniform initialization strategy calculates $limit = \sqrt{\frac{6}{input\_size}}$, assigning initial parameters evenly around zero.

---

## Project Structure

```text
multilayer-perceptron/
├── data/                    # Directory containing the dataset
│   └── data.csv             # The 30-feature dataset (Features starting at col index 2, Labels at col index 1)
├── model/                   # Directory to save trained models and history
│   ├── best_model.npy       # Saved model weights
│   └── history_*.json       # Training loss history (JSON format)
├── src/                     # Core Neural Network library
│   ├── activations.py       # Activation functions (Sigmoid, Softmax, etc.)
│   ├── layers.py            # Implementation of Dense neural layers
│   ├── loss.py              # Loss functions (e.g., Categorical Cross-Entropy, BCE)
│   ├── network.py           # Core Model class handling forward/backward passes and the training loop
│   ├── plot_history.py      # Utilities for plotting training history
│   └── utils.py             # Utility functions for saving/loading model weights and history
├── train.py                 # Script to train the Multilayer Perceptron
├── predict.py               # Script to load the trained model and evaluate on the test/validation set
├── split_data.py            # Script for data loading, standard scaling, and train/test splitting
├── requirements.txt         # Project dependencies
└── README.md                # This file
```

## Installation

1. Clone or download this repository.
2. Initialize and activate a virtual environment (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Preparing the Data
Ensure that your dataset (`data.csv`) is placed inside the `data/` directory.

### 2. Training the Model

You can train the model using `train.py`. The script accepts various command-line arguments to let you construct your own neural network architecture dynamically.

```bash
python train.py --layers 24 24 24 --epochs 84 --loss categoricalCrossentropy --batch_size 8 --learning_rate 0.0314 --optimizer sgd
```

**Available Arguments:**
- `--layers`: List of integers representing units in each hidden layer. Default: `[24, 24, 24]`.
- `--epochs`: Number of epochs for training. Default: `84`.
- `--loss`: The loss function to use (e.g., `categoricalCrossentropy`). Default: `categoricalCrossentropy`.
- `--batch_size`: The batch size for mini-batch gradient descent. Default: `8`.
- `--learning_rate`: The learning rate for the optimizer. Default: `0.0314`.
- `--optimizer`: The optimization algorithm to use (`sgd` or `rmsprop`). Default: `sgd`.

Training the model will output progress over the epochs and automatically save the trained weights and training history into the `model/` directory. Early-stopping will halt the system dynamically when conditions are breached.

### 3. Evaluating the Model

Once trained, you can evaluate the model's performance on the validation set using `predict.py`. The script loads the saved weights (`model/best_model.npy`) and reconstructs the network.

```bash
python predict.py
```

**Expected Output during prediction:**
The `predict.py` script will compute and display:
- Final Binary Cross-Entropy (BCE) Loss
- Precision
- Recall
- F1-Score
- A generated Confusion Matrix
- Overall Accuracy percentage
