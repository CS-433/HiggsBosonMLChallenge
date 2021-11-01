import numpy as np
from helpers.costs import *

def least_squares(y, tx):
    
    """calculate the least squares solution and return optimal weights vector as well as mse"""
   
    x_t = np.transpose(tx)
    
    one = np.dot(x_t,tx)
    two = np.dot(x_t,y)
    w = np.linalg.solve(one,two)
    
    return w, compute_squared_loss(y,tx,w)