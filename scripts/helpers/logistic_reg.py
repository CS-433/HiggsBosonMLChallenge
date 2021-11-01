from helpers.costs import *
import numpy as np


def sigmoid(t):
    """computes the return of the sigmoid function, used for the logistic regression"""
    val= 1/(1 + np.exp(-t))
    if np.all(val) < 0.00001:
        val = 0.01
    if np.all(val) > 0.9999999:
        val = 0.99
    return val


def compute_loss_log_reg(y, tx, w):
    """ computes the loss of the logistic regression, given the weight w, the training tx, and the labels y"""
    if np.all(y) == -1:
        y = 0
    return -np.sum(y * np.log(sigmoid(tx @ w)) + (1 - y) * np.log(1 - sigmoid(tx @ w)))


def calculate_gradient_log_reg(y, tx, w):
    """computes the gradient of loss."""
    prod = np.dot(tx,w)
    x_t = np.transpose(tx)
    return np.dot(x_t, sigmoid(prod)-y)


def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    loss = compute_loss_log_reg(y, tx, w)
    grad = calculate_gradient_log_reg(y, tx, w)
    tmp = gamma * grad
    w = w - tmp.reshape((31, 1))

    return loss, w


def logistic_reg(y, tx, initial_w, max_iter, gamma):
    """main function for logistic regression, performed by doing the gradient descent"""
    tx = np.c_[np.ones((y.shape[0], 1)), tx]
    w = np.zeros((tx.shape[1], 1))
    loss = 0

    for iter in range(max_iter):
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)

    return w, loss


def calculate_loss_reg_logistic_regression(y, tx, w, lambda_):
    """compute the cost by negative log likelihood for regularized logistic regression, with lambda as a penalty parameter"""
    # Here we use lambda as defined as in class !
    return compute_loss_log_reg(y, tx, w) + lambda_ / 2 * np.dot(np.transpose(w), w)


def calculate_gradient_reg_logistic_regression(y, tx, w, lambda_):
    """compute the gradient of the regularized loss, with lambda as a regularizing parameter."""
    # Here we use lambda as defined as in class !
    return calculate_gradient_log_reg(y, tx, w).reshape((31, 1)) + lambda_ * w


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """implements regularized logistic regression using gradient descent."""
    w = initial_w
    tx = np.c_[np.ones((y.shape[0], 1)), tx]

    # start the logistic regression
    for iter in range(max_iters):
        # get loss, gradient and update w.
        gradient = calculate_gradient_reg_logistic_regression(y, tx, w, lambda_)
        w = w - gamma * gradient

    loss = calculate_loss_reg_logistic_regression(y, tx, w, lambda_)

    return w, loss
