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
In each `Dense` layer (`src/layers.py`), the inputs undergo a linear transformation followed by a non-linear activation function. For a layer $l$ and a batch of $m$ examples:

$$ Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]} $$
$$ A^{[l]} = g(Z^{[l]}) $$

Where:
- $W^{[l]}$ is the weight matrix of shape $(n^{[l]}, n^{[l-1]})$.
- $A^{[l-1]}$ is the input matrix (or activations from the previous layer) of shape $(n^{[l-1]}, m)$.
- $b^{[l]}$ is the bias vector of shape $(n^{[l]}, 1)$.
- $g$ is the activation function.

**Activations (`src/activations.py`)**: 
- **Sigmoid**: Used for hidden layers to introduce non-linearity, mapping values to $(0, 1)$.
  $$ \sigma(Z) = \frac{1}{1 + e^{-Z}} $$
  *Implementation detail*: $Z$ is dynamically clipped $[-500, 500]$ to prevent exponential overflow (`RuntimeWarning`).
- **Softmax**: Used at the output layer for multi-class/binary probability distribution.
  $$ \text{Softmax}(Z_i) = \frac{e^{Z_i}}{\sum_{j} e^{Z_j}} $$
  *Implementation detail*: To ensure numerical stability, $\max(Z)$ across the vector is subtracted before exponentiation ($e^{Z_i - \max(Z)}$), preventing `NaN` generation from massive scalar evaluation.

### 2. Loss Functions (`src/loss.py`)
Loss functions quantify the difference between the network's predictions $\hat{y}$ (derived from $A^{[L]}$) and the true labels $y$.
- **Categorical Cross Entropy (CCE)**: Used during the core training loop for multi-class formatted inputs (e.g., one-hot encoded `[1,0]` vs `[0,1]`). 
  Summing across classes $C$ and averaging across $m$ batch size:
  $$ L_{CCE} = - \frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{C} y^{(i)}_k \log(\hat{y}^{(i)}_k) $$
- **Binary Cross Entropy (BCE)**: Applied strictly on validation checks to test binary correctness. 
  $$ L_{BCE} = - \frac{1}{m} \sum_{i=1}^{m} \left( y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right) $$
  *Implementation detail*: Probabilities $\hat{y}$ are clipped against a minimal threshold ($\epsilon = 1e-15$) bounding them between $[1e-15, 1 - 1e-15]$ to prevent computing $\log(0)$.

### 3. Backpropagation
Backpropagation relies on the chain rule of calculus to compute the gradient of the loss function with respect to each weight $W$ and bias $b$ in the network.
Starting from the output layer's error gradient $dZ^{[L]}$ (which simplifies to $A^{[L]} - y$ when using Cross-Entropy combined with Softmax):
$$ dZ^{[L]} = A^{[L]} - y $$

For any preceding layer $l$, the gradients are propagated backwards:
- **Weight Gradients**: The partial derivative of the loss regarding weights.
  $$ dW^{[l]} = \frac{\partial L}{\partial W^{[l]}} = \frac{1}{m} dZ^{[l]} \cdot (A^{[l-1]})^T $$
- **Bias Gradients**: The partial derivative of the loss regarding biases (summed across the batch dimension).
  $$ dB^{[l]} = \frac{\partial L}{\partial b^{[l]}} = \frac{1}{m} \sum_{i=1}^{m} dZ^{[l](i)} $$
- **Input Error Propagation**: The error pushed to the previous layer $l-1$ to continue the chain.
  $$ dZ^{[l-1]} = (W^{[l]})^T \cdot dZ^{[l]} * g'(Z^{[l-1]}) $$
  *(Note: Specific implementations vary depending on optimizer integrations, but the underlying chain geometry remains identical).*

### 4. Optimization Algorithms
Weights are adjusted iteratively to minimize loss. This repository supports two forms of gradient descent methodologies:
- **Standard SGD (Stochastic Gradient Descent)**: 
  Updates weights scaling linearly by the learning rate $\alpha$.
  $$ W^{[l]} = W^{[l]} - \alpha \cdot dW^{[l]} $$
  $$ b^{[l]} = b^{[l]} - \alpha \cdot dB^{[l]} $$

- **RMSprop (Root Mean Square Propagation)**:
  An adaptive learning rate method designed to resolve diminishing/exploding gradient problems. In RMSprop, we maintain an exponentially decaying average of squared gradients:
  $$ v_{dW} = \beta v_{dW} + (1 - \beta) (dW^{[l]})^2 $$
  $$ v_{dB} = \beta v_{dB} + (1 - \beta) (dB^{[l]})^2 $$
  Weights are updated inversely proportional to the square root of this moving average, standardizing the variance over steps. Epsilon ($\epsilon = 1e-8$) is added to prevent zero-division:
  $$ W^{[l]} = W^{[l]} - \frac{\alpha}{\sqrt{v_{dW}} + \epsilon} dW^{[l]} $$
  $$ b^{[l]} = b^{[l]} - \frac{\alpha}{\sqrt{v_{dB}} + \epsilon} dB^{[l]} $$
  *(Algorithm default: $\beta = 0.9$)*

### 5. Training Mechanics (`src/network.py`)
- **Mini-Batch Gradient Descent**: During each epoch, datasets are stochastically shattered utilizing `np.random.permutation()` and processed in sub-chunks of `$batch\_size$`. This injects beneficial noise into the learning matrix and enables computational speed scaling over strictly Full-Batch processes.
- **Early Stopping**: Validation loss is tracked dynamically per epoch. If the validation loss fails to decrease over $patience$ consecutive epochs ($Patience = 10$), the model assumes local minima convergence. The best historical parameters (`best_model.npy`) are serialized to disk, terminating the loop safely.
- **Weight Initializations (He Uniform)**: Weight parameter initialization prevents vanishing gradients out of the gate. For an array with $n^{[l-1]}$ incoming inputs (`input_size`), weights $W^{[l]}$ are populated from a uniform distribution bounded by:
  $$ \text{limit} = \sqrt{\frac{6}{n^{[l-1]}}} $$
  $$ W \sim \mathcal{U}(-\text{limit}, \text{limit}) $$

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
