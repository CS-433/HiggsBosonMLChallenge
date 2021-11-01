from helpers.gradient_descent import gradient_descent
from helpers.stochastic_gradient_descent import stochastic_gradient_descent
from helpers.ridge import ridge
from helpers.logistic_reg import *

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
  return gradient_descent(y,tx,initial_w,max_iters,gamma)

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
  return stochastic_gradient_descent(y,tx,initial_w,max_iters,gamma)

def least_squares(y, tx):
  return least_squares(y,tx)
  
def ridge_regression(y, tx, lambda_):
  return ridge(y,tx,lambda_)   

def logistic_regression(y, tx, initial_w,
max_iters, gamma):
  return logistic_reg(y,tx,initial_w,max_iters,gamma)

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
  return reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)
