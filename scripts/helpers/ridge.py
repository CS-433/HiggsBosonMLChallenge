import numpy as np
from helpers.costs import *

def ridge(y, tx, lambda_):
    """implement ridge regression."""
    x_t = np.transpose(tx)
    N = tx.shape[0]
    K = tx.shape[1]
    Id = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    one = np.dot(x_t,tx) + Id
    two = np.dot(x_t,y)
    
    w = np.linalg.solve(one,two)
    loss = compute_squared_loss(y,tx,w)
    
    return w,loss 
