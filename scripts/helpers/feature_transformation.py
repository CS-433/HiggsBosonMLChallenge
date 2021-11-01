import numpy as np
"""Different methods that are needed to transform our features in order to improve accuracy"""
def poly_expansion(x,degree):
    """Polynomial feature expansion: Takes a matrix X and returns a new matrix (1,X,X^2,...,X^degree)"""
    poly = np.ones((len(x),1))
    degree = int(degree)
    for deg in range(1,degree+1):
        poly = np.c_[poly,np.power(x,deg)]
        
    #add interaction terms
    #for one in range(x.shape[1]):
    #    for two in range(x.shape[1]):
    #        if one>=two:
    #            column = (x[:,one]*x[:,two]).reshape((x.shape[0],1))
    #            poly = np.c_[poly,column]"""

    return poly

def log_transform(x, features):
    """Apply a log transform to all features of x that have been specified in the arguments. Returns a new matrix and doesn't modify the given x"""
    #For all the features that have negative values, we take their opposite in order to give only positive values to the log-function
    new = np.copy(x)
    for f in features:
        column = x[:,f]
        column = np.abs(column)
        column = np.log(1+column)
        
        new[:,f] = column
        
    return new   
    
    

