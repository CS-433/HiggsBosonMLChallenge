from helpers.costs import *
from helpers.sigmoid import *
import numpy as np

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    prod = np.dot(tx,w)
    x_t = np.transpose(tx)
    return np.dot(x_t, sigmoid(prod)-y)


def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    loss = calculate_loss(y,tx,w)
    grad = calculate_gradient(y,tx,w)
    w -= gamma*grad
  
    return loss, w


def logistic_reg(y, tx,initial_w,max_iter,gamma):
    
    tx = np.c_[np.ones((y.shape[0], 1)), tx]
    w = np.zeros((tx.shape[1], 1))
    loss = 0

    for iter in range(max_iter):

      loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        
    return w,loss
   