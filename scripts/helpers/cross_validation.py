# -*- coding: utf-8 -*-
"""Cross Validation"""

import numpy as np
import matplotlib.pyplot as plt


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def split_validation(y, x, k_indices, k):
    """return the loss of ridge regression."""
    # get k'th subgroup in test, others in train

    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)

    y_te = y[te_indice]
    y_tr = y[tr_indice]
    x_te = x[te_indice]
    x_tr = x[tr_indice]
    return x_tr, y_tr, x_te,y_te


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