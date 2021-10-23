# -*- coding: utf-8 -*-
"""Function used to compute the loss."""
import numpy as np
import matplotlib.pyplot as plt

def compute_squared_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
   
    # w use mse
    # in the theoretical part of the session we have derived that L(w) = 1/2N e'*e
    
    N = y.shape[0]
    e = y- np.dot(tx,w)
    
    return (1/2*N) * np.sum(e*e) 
