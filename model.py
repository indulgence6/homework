import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def logsumexp(x, axis=1, keepdims=True):
    x_max = np.max(x, axis=axis, keepdims=keepdims)
    return x_max + np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=keepdims))


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, activation='relu'):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros(output_size)
        self.activation = activation
        
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        if self.activation == 'relu':
            self.a1 = np.maximum(0, self.z1)
        elif self.activation == 'sigmoid':
            self.a1 = 1 / (1 + np.exp(-self.z1))
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.z2
    
    def compute_loss(self, logits, y, lambda_reg):
        m = y.shape[0]
        log_probs = logits - logsumexp(logits, axis=1, keepdims=True)
        loss = -np.mean(log_probs[np.arange(m), y.astype(int)])
        reg_loss = 0.5 * lambda_reg * (np.sum(self.W1**2) + np.sum(self.W2**2))
        return loss + reg_loss
    
    def backward(self, X, y, lambda_reg):
        m = y.shape[0]
        probs = softmax(self.z2)
        probs[range(m), y] -= 1
        d_z2 = probs / m
        
        d_W2 = np.dot(self.a1.T, d_z2) + lambda_reg * self.W2
        d_b2 = np.sum(d_z2, axis=0)
        
        d_a1 = np.dot(d_z2, self.W2.T)
        if self.activation == 'relu':
            d_z1 = d_a1 * (self.z1 > 0)
        elif self.activation == 'sigmoid':
            sig = 1 / (1 + np.exp(-self.z1))
            d_z1 = d_a1 * sig * (1 - sig)
        
        d_W1 = np.dot(X.T, d_z1) + lambda_reg * self.W1
        d_b1 = np.sum(d_z1, axis=0)
        
        return d_W1, d_b1, d_W2, d_b2

def accuracy(logits, y):
    return np.mean(np.argmax(logits, axis=1) == y)