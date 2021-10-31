import numpy as np
#Polynomial feature expansion: Takes a matrix X and returns a new matrix (1,X,X^2,...,X^n)
def poly_expansion(x,degree):
    poly = np.ones((len(x),1))
    degree = int(degree)
    for deg in range(1,degree+1):
        poly = np.c_[poly,np.power(x,deg)]
    return poly

