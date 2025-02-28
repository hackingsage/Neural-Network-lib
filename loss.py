import numpy as np

class CategoricalCrossEntropyLoss:
    '''
    L(y_true,y_pred) = -sum[i=1->N](sum[j=1->C](y_true*log(y_pred)))
    N: Number of samples in the batch
    C: Number of classes
    y_true is the true label
    y_pred is the predicted probability
    '''
    def __init__(self,epsilon=1e-15):
        self.epsilon = epsilon # Prevents taking log(0)
        self.pred_prob = None # To store the predicted probabilites during forward pass
        self.true_labels = None # Store true labels for backward pass
    
    def forward(self,pred_probs,true_labels):
        pred_probs = np.clip(pred_probs, self.epsilon, 1 - self.epsilon) #Clip predicted probabilites to avoid log(0)
        self.true_labels = true_labels
        self.pred_prob = pred_probs
        loss = -np.sum(true_labels * np.log(pred_probs)) / true_labels.shape[0]
        return loss
    
    def backward(self):
        grad = -self.true_labels / self.pred_prob
        grad /= self.true_labels.shape[0]
        return grad

class MSE:
    def __init__(self):
        self.true_values = None
        self.pred_values = None

    def forward(self,pred_value,true_values):
        self.true_values = true_values
        self.pred_values = pred_value
        loss = np.mean((true_values - pred_value) ** 2)
        return loss
    
    def backward(self):
        grad = 2 * (self.pred_values - self.true_values) / self.true_values.shape[0]
        return grad

class MAE:
    def __init__(self):
        self.true_values = None
        self.pred_values = None
    
    def forward(self,pred_values,true_values):
        self.true_values = true_values
        self.pred_values = pred_values
        loss = np.mean(np.abs(true_values - pred_values))
        return loss
    
    def backward(self):
        grad = np.sign(self.pred_values - self.true_values) / self.true_values.shape[0]
        return grad

class CrossEntropyLoss:
    '''
    L(y_true,y_pred) = -sum(y_true * log(y_pred))
    '''
    def __init__(self,epsilon=1e-15):
        self.epsilon = epsilon
        self.pred_probs = None
        self.true_labels = None

    def forward(self,pred_probs,true_labels):
        '''
        pred_probs = Predicted probabilites (shape: [batch_size, num_classes])
        true_labels = True Labels (shape: [batch_size, num_classes])
        '''
        pred_probs = np.clip(pred_probs, self.epsilon, 1 - self.epsilon) #Prevent log(0)
        self.true_labels = true_labels
        self.pred_probs = pred_probs
        loss = -np.sum(true_labels * np.log(pred_probs)) / true_labels.shape[0]
        return loss
    
    def backward(self):
        grad = -self.true_labels / self.pred_probs
        grad /= self.true_labels.shape[0]
        return grad