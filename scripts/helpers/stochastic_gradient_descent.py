import numpy as np
import matplotlib.pyplot as plt
from helpers.costs import *
from helpers.proj1_helpers import batch_iter

def compute_stoch_gradient(y, tx, w):
    """Computes one step of stochastic gradient descent"""
    e = y - np.dot(tx,w)
    N = y.shape[0]
    return (-1/N)* np.dot(np.transpose(tx),e)


def stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma,  batch_size = 1 ):
    """Stochastic Gradient Descent for the given parameters and batch_size, returns optimal estimates for the weight vector"""
    w = initial_w
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            grad = compute_stoch_gradient(minibatch_y,minibatch_tx,w)
            
            w = w - gamma * grad
            
    loss = compute_squared_loss(minibatch_y,minibatch_tx,w)        
    return w, loss
