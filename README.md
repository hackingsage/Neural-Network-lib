# My Neural Network Library

A lightweight neural network library implemented from scratch using NumPy. Supports fully connected layers, activation functions, loss functions, and optimizers.

## Features
- Dense layers with configurable activation functions
- Activation functions: ReLU, Leaky ReLU, Sigmoid, Tanh, Softmax, Linear
- Loss functions: Mean Squared Error (MSE), Categorical Cross-Entropy
- Optimizers: Stochastic Gradient Descent (SGD), Adam
- Training loop with backpropagation
- Example scripts for XOR and Iris dataset classification

## Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/hackingsage/Neural-Network-lib.git
cd Neural-Network0-lib
pip install -r requirements.txt
```

## Usage

### Example 1: XOR Dataset
Train a simple neural network on the XOR problem.

```python
from nnlib import NeuralNetwork
from layer import DenseLayer
from loss import MSE
from optimizers import SGDOptimizer
import numpy as np

# XOR dataset
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]])

# Create and configure the network
nn = NeuralNetwork()
nn.add_layer(DenseLayer(2, 4, 'relu'))
nn.add_layer(DenseLayer(4, 1, 'sigmoid'))
nn.set_loss(MSE())
nn.set_optimizer(SGDOptimizer({'W_0': nn.layers[0].weights, 'b_0': nn.layers[0].biases, 'W_1': nn.layers[1].weights, 'b_1': nn.layers[1].biases}, lr=0.1))

# Train and predict
nn.train(X_train, y_train, epochs=100, batch_size=2)
print("Predictions:", nn.predict(X_train))
```

### Example 2: Iris Dataset Classification
Classify Iris dataset using Softmax activation and Categorical Cross-Entropy Loss.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from nnlib import NeuralNetwork
from layer import DenseLayer
from loss import CategoricalCrossEntropyLoss
from optimizers import SGDOptimizer
import numpy as np

# Load and preprocess dataset
iris = load_iris()
X = iris.data
y = iris.target
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y.reshape(-1, 1))
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_onehot, test_size=0.2, random_state=42)

# Define the network
nn = NeuralNetwork()
nn.add_layer(DenseLayer(4, 8, 'relu'))
nn.add_layer(DenseLayer(8, 3, 'softmax'))
nn.set_loss(CategoricalCrossEntropyLoss())
nn.set_optimizer(SGDOptimizer({'W_0': nn.layers[0].weights, 'b_0': nn.layers[0].biases, 'W_1': nn.layers[1].weights, 'b_1': nn.layers[1].biases}, lr=0.01))

# Train and evaluate
nn.train(X_train, y_train, epochs=100, batch_size=16)
accuracy = np.mean(np.argmax(nn.predict(X_test), axis=1) == np.argmax(y_test, axis=1))
print(f"Test Accuracy: {accuracy:.4f}")
```

## File Structure
```
my_neural_network_lib/
│── nnlib.py               # Main Neural Network library
│── layer.py               # Implementation of Dense layers
│── activations.py         # Activation functions
│── loss.py                # Loss functions
│── optimizers.py          # Optimizers (SGD, Adam)
│── example.py             # Example: XOR dataset
│── example_iris.py        # Example: Iris dataset classification
│── requirements.txt       # Dependencies (NumPy, scikit-learn)
│── README.md              # Documentation
│── .gitignore             # Ignore unnecessary files
```

## License
This project is released under the MIT License.

