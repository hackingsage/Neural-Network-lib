import numpy as np
from nnlib import NeuralNetwork
from layer import DenseLayer
from loss import MSE
from optimizers import SGDOptimizer

# XOR dataset
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]])

# Create the network
nn = NeuralNetwork()
nn.add_layer(DenseLayer(2, 4, 'relu'))
nn.add_layer(DenseLayer(4, 1, 'sigmoid'))  # Ensure 'sigmoid' is a string
nn.set_loss(MSE())

# Set optimizer with layer parameters
params = {
    'W_0': nn.layers[0].weights,
    'b_0': nn.layers[0].biases,
    'W_1': nn.layers[1].weights,
    'b_1': nn.layers[1].biases
}
nn.set_optimizer(SGDOptimizer(params, lr=0.1))

# Train the network
nn.train(X_train, y_train, epochs=100, batch_size=2)

# Make predictions
predictions = nn.predict(X_train)
print("Predictions:", predictions)