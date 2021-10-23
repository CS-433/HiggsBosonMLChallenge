import numpy as np
import matplotlib.pyplot as plt
from costs import *
from proj1_helpers import batch_iter

def compute_stoch_gradient(y, tx, w):
        e = y - np.dot(tx,w)
        N = y.shape[0]
        return (-1/N)* np.dot(np.transpose(tx),e)


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
        w = initial_w
        for n_iter in range(max_iters):
            for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
                grad = compute_stoch_gradient(minibatch_y,minibatch_tx,w)
                
                w = w - gamma * grad
                
        loss = compute_squared_loss(minibatch_y,minibatch_tx,w)        
        return w, loss