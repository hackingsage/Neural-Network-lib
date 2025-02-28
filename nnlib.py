import numpy as np
from optimizers import *

class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss_fn = None
        self.optimizer = None

    def add_layer(self, layer):
        self.layers.append(layer)

    def set_loss(self, loss_fn):
        self.loss_fn = loss_fn

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def forward(self, inputs):
        output = inputs
        for i, layer in enumerate(self.layers):
            output = layer.forward(output)
        return output

    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)

    def train(self, X_train, y_train, epochs=100, batch_size=32):
        num_samples = X_train.shape[0]
        for epoch in range(epochs):
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_train = X_train[indices]
            y_train = y_train[indices]

            for i in range(0, num_samples, batch_size):
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]

                predictions = self.forward(X_batch)
                loss = self.loss_fn.forward(predictions, y_batch)

                grad_output = self.loss_fn.backward()
                self.backward(grad_output)

                grads = {
                    f'W_{i}': layer.grad_weights for i, layer in enumerate(self.layers)
                }
                grads.update({
                    f'b_{i}': layer.grad_biases for i, layer in enumerate(self.layers)
                })
                params = {
                    f'W_{i}': layer.weights for i, layer in enumerate(self.layers)
                }
                params.update({
                    f'b_{i}': layer.biases for i, layer in enumerate(self.layers)
                })
                if self.optimizer is None:
                    self.optimizer = SGDOptimizer(params, lr=0.01)
                self.optimizer.step(grads)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X_test):
        return self.forward(X_test)