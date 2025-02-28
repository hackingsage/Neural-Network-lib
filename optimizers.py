import numpy as np

class AdamOptimizer:
    def __init__(self,params,lr=0.001,beta1=0.9,beta2=0.999,epsilon=1e-8):
        """
        params: Dictionary of parameters to optimize
        lr: Learning rate
        beta1: Exponential Decay rate for the first moment estimates
        beta2: Exponential Decay rate for the second moment estimates
        epsilon: Small constant to avoid division by zero
        """
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {k : np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0 # Timestop

    def step(self,grads):
        '''
        First moment (mean) m_t = beta1 * m_t-1 + (1 - beta1) * grad
        Second moment (uncentered variance) v_t = beta2 * v_t-1 + (1 - beta2) * (grad ** 2)
        Bias-corrected estimates:
            m_hat = m_t / (1 - beta1 ** 2)
            v_hat = v_t / (1 - beta2 ** 2)
        The parameter update:
            param_t = param_t-1 - lr * [m_hat / (sqrt(v_hat) + epsilon)]
        '''
        self.t += 1
        for key in self.params:
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            self.params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

class SGDOptimizer:
    def __init__(self,params,lr=0.01):
        self.params = params
        self.lr = lr
    
    def step(self,grads):
        '''
        param_t = param_t-1 - lr * grad(param)
        '''
        for key in self.params:
            self.params[key] -= self.lr * grads[key]