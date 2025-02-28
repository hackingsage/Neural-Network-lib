import numpy as np
from activations import *

class DenseLayer:
    def __init__(self,n_inputs,n_neurons,activation_func):
        self.weights = self.initialize_weights(n_inputs,n_neurons,activation_func)
        self.biases = np.zeros((1,n_neurons))
        self.activation_func = self.get_activation(activation_func)
        self.inputs = None
        self.output = None
        if not hasattr(self.activation_func, 'forward'):
            raise ValueError(f"Activation {activation_func} does not have a 'forward' method")

    def forward(self,inputs):
        self.inputs = inputs
        self.output = np.dot(inputs,self.weights) + self.biases
        return self.activation_func.forward(self.output)

    def get_activation(self, activation_func):
        activation_map = {
            'relu': ReLu(),
            'leaky_relu': Leaky_ReLU(),
            'sigmoid': Sigmoid(),
            'tanh': Tanh(),
            'linear': Linear(),
            'softmax': SoftMax()
        }
        if activation_func not in activation_map:
            raise ValueError(f"Unsupported activation function: {activation_func}")
        return activation_map[activation_func]

    def initialize_weights(self,n_inputs,n_neurons,activation_func):
        '''
        In neural networks, the initial values of weights can significantly impact the training process. 
        Poor initialization can lead to:
        1. Vanishing Gradients: If weights are too small, the gradients during backpropagation can become extremely small, 
                                slowing down or stopping learning.
        2. Exploding Gradients: If weights are too large, the gradients can become extremely large, 
                                causing unstable updates and divergence.
        3. Slow Convergence: Poor initialization can make the network take longer to converge or get stuck in poor local minima.
        '''
        if activation_func in ['relu','leaky_relu']:
            return np.random.rand(n_inputs,n_neurons) * np.sqrt(2/n_inputs) # He Initialization
        elif activation_func in ['sigmoid','tanh','linear','softmax']:
            return np.random.rand(n_inputs,n_neurons) * np.sqrt(1/n_inputs) # Xavier Initialization
        else:
            raise ValueError(f"Unsupported activation function: {activation_func}")
    
    def backward(self,grad_output):
        grad_activation = self.activation_func.backward(grad_output)
        self.grad_weights = np.dot(self.inputs.T,grad_activation)
        self.grad_biases = np.sum(grad_activation,axis=0,keepdims=True)
        grad_inputs = np.dot(grad_activation, self.weights.T)
        return grad_inputs
    