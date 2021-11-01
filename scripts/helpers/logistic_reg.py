from helpers.costs import *
from helpers.sigmoid import *
import numpy as np

def compute_gradient_log_reg(y, tx, w):
    """compute the gradient of loss."""
    prod = np.dot(tx,w)
    x_t = np.transpose(tx)
    return np.dot(x_t, sigmoid(prod)-y)


def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    loss=compute_loss_log_reg(y, tx,w)
    grad = compute_gradient_log_reg(y,tx,w)
    tmp = gamma*grad
    w = w - tmp.reshape((31,1))
    
    return loss, w

def logistic_reg(y, tx,initial_w,max_iter,gamma):
    """Does logistic regression for max_iter steps and returns the optimal weight vector as well as the loss"""
    tx = np.c_[np.ones((y.shape[0], 1)), tx]
    w  = np.zeros((tx.shape[1], 1))
    loss = 0

    for iter in range(max_iter):
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
    
    return w,loss