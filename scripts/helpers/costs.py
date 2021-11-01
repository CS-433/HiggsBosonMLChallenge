# -*- coding: utf-8 -*-
"""Functions used to compute the loss"""
import numpy as np
import matplotlib.pyplot as plt

def compute_squared_loss(y, tx, w):
    """compute mean squared loss given the expected labels, the datasets and the weight vector"""
  
    
    N = y.shape[0]
    e = y- np.dot(tx,w)
    
    return 1/2 * np.mean(e**2) 

def compute_log_likelihood(y,tx,w):
    """compute negative log likelihood given the expected label, the dataset and the weight vector"""
    product = np.dot(tx,w)
    exp = np.exp(product)
    log = np.log(exp+1)
    one = y * product
    diff = log - one
    return sum(diff)