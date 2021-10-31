# -*- coding: utf-8 -*-
"""Cross Validation"""

import numpy as np
import matplotlib.pyplot as plt
from helpers.feature_transformation import *
from helpers.ridge import *

def build_k_indices(y, k_fold, seed):
    """get indices of k-subgroups of our dataset and our labels"""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def split_validation(y, x, k_indices, k):
    """build training set and test set according to the k value in our cross validation"""

    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)

    y_te = y[te_indice]
    y_tr = y[tr_indice]
    x_te = x[te_indice]
    x_tr = x[tr_indice]
    return x_tr, y_tr, x_te,y_te

def cartesian_product(a,b):
    """join two arrays together by doing a cartesian product"""
    return np.transpose([np.tile(a,len(b)),np.repeat(b,len(a))])

def cross_validation_ridge(k_fold,tX,labels,seed=1):
    """grid-search cross validation to find the best hyperparameters (degree and lambda) for ridge regression with a polynomial     expansion """
    degrees  = np.arange(10)
    lambdas = np.logspace(-7,0,8)
    degrees_plus_lambdas = cartesian_product(degrees,lambdas)
    
    k_indices = build_k_indices(labels,10,seed)
    
    rmse_tr = []
    rmse_te = []

    best = 2**30
    best_degree = -1
    best_lambda = -1
    
    for pair in degrees_plus_lambdas:
        degree = pair[0]
        lambda_ = pair[1]

        rmse_tr_temp = []
        rmse_te_temp = []
    
          
        for k in range(k_fold):
            x_tr,y_tr,x_te,y_te = split_validation(labels,tX,k_indices,k)

            x_tr = poly_expansion(x_tr,degree)
            x_te = poly_expansion(x_te,degree)

            weights, _ = ridge(y_tr, x_tr, lambda_)
            #weights_te, loss_te = ridge(y_te, x_te, lambda_)

            loss_tr = np.sqrt(2*compute_squared_loss(y_tr,x_tr,weights))
            loss_te = np.sqrt(2*compute_squared_loss(y_te,x_te,weights))

            rmse_tr_temp.append(loss_tr)
            rmse_te_temp.append(loss_te)

        rmse_tr_mean = np.mean(rmse_tr_temp)
        rmse_te_mean = np.mean(rmse_te_temp)

        if(rmse_te_mean < best):
            best = rmse_te_mean
            best_degree = degree
            best_lambda = lambda_
        
        print("Degree:",degree," Lambda:",lambda_," RMSE Training:",rmse_tr_mean, " RMSE Test:", rmse_te_mean)
        rmse_tr.append([degree,lambda_,rmse_tr_mean])
        rmse_te.append([degree,lambda_, rmse_te_mean])
    
    return [best,best_degree,best_lambda,np.asarray(rmse_tr),np.asarray(rmse_te)]




def cross_validation_log_visualization(lambds, mse_tr, mse_te):
    """visualization the curves of mse_tr and mse_te."""
    plt.semilogx(lambds, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("rmse")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")
    
def cross_validation_visualization(degrees, mse_tr, mse_te):
    """visualization the curves of mse_tr and mse_te."""
    #plt.semilogx(lambds, mse_tr, marker=".", color='b', label='train error')
    #plt.semilogx(lambds, mse_te, marker=".", color='r', label='test error')
    #plt.scatter(degrees,mse_tr)
    plt.scatter(degrees,mse_te)
    plt.xlabel("degree")
    plt.ylabel("rmse")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")   
    plt.show() 