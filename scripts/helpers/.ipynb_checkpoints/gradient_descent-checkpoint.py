# -*- coding: utf-8 -*-
"""Gradient Descent"""

import numpy as np
import matplotlib.pyplot as plt
from costs import *


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    e = y - np.dot(tx,w)
    N = y.shape[0]
    return (-1/N)* np.dot(np.transpose(tx),e)


def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    w = initial_w
    for n_iter in range(max_iters):
        
        grad = compute_gradient(y,tx,w)
        
        w = w - gamma * grad 
     
        # store w and loss

    loss = compute_squared_loss(y,tx,w)
  
    return w, loss
