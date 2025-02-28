import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from nnlib import NeuralNetwork
from layer import DenseLayer
from loss import CategoricalCrossEntropyLoss
from optimizers import SGDOptimizer

# Load the Iris dataset
iris = load_iris()
X = iris.data  # 150 samples, 4 features
y = iris.target  # 3 classes (0, 1, 2)

# Preprocess the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Normalize features

# One-hot encode the targets (3 classes)
encoder = OneHotEncoder(sparse_output=False)  # Use sparse_output instead of sparse
y_onehot = encoder.fit_transform(y.reshape(-1, 1))

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_onehot, test_size=0.2, random_state=42)

# Create the neural network
nn = NeuralNetwork()
nn.add_layer(DenseLayer(4, 8, 'relu'))  # Input layer (4 features) → Hidden layer (8 neurons), ReLU
nn.add_layer(DenseLayer(8, 3, 'softmax'))  # Hidden layer → Output layer (3 classes), SoftMax

# Set loss and optimizer
nn.set_loss(CategoricalCrossEntropyLoss())
params = {
    'W_0': nn.layers[0].weights,
    'b_0': nn.layers[0].biases,
    'W_1': nn.layers[1].weights,
    'b_1': nn.layers[1].biases
}
nn.set_optimizer(SGDOptimizer(params, lr=0.01))

# Train the network
nn.train(X_train, y_train, epochs=100, batch_size=16)

# Make predictions on test set
predictions = nn.predict(X_test)
print("Predictions (probabilities):", predictions)

# Convert probabilities to class predictions (argmax)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)
accuracy = np.mean(predicted_classes == true_classes)
print(f"Accuracy on test set: {accuracy:.4f}")