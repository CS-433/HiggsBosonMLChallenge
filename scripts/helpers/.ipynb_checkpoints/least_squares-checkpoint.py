import numpy as np

def least_squares(y, tx):
    
    """calculate the least squares solution."""
   
    # returns mse, and optimal weights
    x_t = np.transpose(tx)
    
    one = np.dot(x_t,tx)
    two = np.dot(x_t,y)
    
    return np.linalg.solve(one,two), compute_squared_loss(y,tx,w)