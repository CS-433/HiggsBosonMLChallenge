import numpy as np

def sigmoid(t):
    """apply the sigmoid function on t."""
    exp = np.exp(-t)
    
    return 1.0/(1+exp)