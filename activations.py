import numpy as np

class SoftMax:
    def __init__(self):
        self.probabilities = None
    
    def forward(self,inputs):
        '''
        z -> inputs Eg:- [2,1,1]
        z_max = max(z) -> 2 
        z_shifted = [z-z_max] -> [0,-1,-1]
        exp(z_shifted) = [1,0,0.3679,0.3679]
        sum(exp(z_shifted)) = 1 + 0.3679 + 0.3679 = 1.735
        softmax(z1) = 1/1.735 = P(z1)
        softmax(z2) = 0.3679/1.735 = P(z2)
        softmax(z3) = 0.3679/1.735 = P(z3)
        '''
        exp_values = np.exp(inputs-np.max(inputs,axis=1,keepdims=True)) # np.exp(inputes_shifted)
        self.probabilities = exp_values / np.sum(exp_values,axis=1,keepdims=True)
        return self.probabilities

    def backward(self,grad_output):
        batch_size, num_classes = self.probabilities.shape
        grad_input = np.zeros_like(grad_output)
        for i in range(batch_size):
            softmax_out = self.probabilities[i].reshape(-1, 1)  # Shape: (num_classes, 1)
            grad_output_i = grad_output[i].reshape(-1, 1)  # Shape: (num_classes, 1)
            jacobian = np.diagflat(softmax_out) - np.dot(softmax_out, softmax_out.T)  # Shape: (num_classes, num_classes)
            grad_input[i] = np.dot(jacobian, grad_output_i).flatten()  # Shape: (num_classes,)
        return grad_input

class Leaky_ReLU:
    def __init__(self,alpha=0.01):
        self.alpha = alpha
        self.inputs = None
    
    def forward(self,inputs):
        '''
        Leaky ReLu Activation function = 
        {
            x < 0 -> f(x) = alpha * x
            x > 0 -> f(x) = x
        }
        '''
        self.inputs = inputs
        return np.where(inputs > 0,inputs,self.alpha*inputs)
    
    def backward(self,grad_output):
        grad_input = grad_output * np.where(self.inputs > 0, 1, self.alpha)
        return grad_input

class ReLu:
    def __init__(self):
        self.inputs = None
    
    def forward(self,inputs):
        '''
        ReLu Activation function = 
        {
            x < 0 -> f(x) = 0;
            x > 0 -> f(x) = x 
        }
        '''
        self.inputs = inputs
        return np.maximum(0,inputs)
    
    def backward(self,grad_ouput):
        return grad_ouput * (self.inputs > 0)

class Sigmoid:
    def __init__(self):
        self.sigmoid = None

    def forward(self,inputs):
        '''
        sigmoid(x) = 1/(1+exp(-x))
        '''
        exp_value = 1/np.exp(inputs)
        self.sigmoid = 1 / (1 + exp_value)
        return self.sigmoid
    
    def backward(self,grad_output):
        return grad_output * self.sigmoid * (1 - self.sigmoid)


class Tanh:
    def __init__(self):
        self.tanh = None
    
    def forward(self,inputs):
        '''
        tanh(x) = (exp(x)-exp(-x))/(exp(x)+exp(-x))
        '''
        self.tanh = np.tanh(inputs)
        return self.tanh
    
    def backward(self,grad_output):
        return grad_output * (1 - self.tanh ** 2)
    
class Linear:
    def forward(self,inputs):
        '''
        f(x) = x
        '''
        return inputs
    
    def backward(self,grad_output):
        return grad_output