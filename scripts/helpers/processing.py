import numpy as np
"""Functions used to to various pre-processing steps to clean our dataset. Everything is nicely put together in the pre_process_data_pipeline"""


def groupy_by_jet_num(x,y):
    """Group the data into three sets, depending on whether their PRI_jet_num is {0,1,(2|3)} and return the created sets => also return the corresponding y's"""
    #create masks to extract each one of the subsets
    mask0 = x[:,22] == 0
    mask1 = x[:,22] == 1
    mask2 = x[:,22] == 2
    mask3 = x[:,22] == 3
    mask2_3 = np.logical_or(mask2,mask3)
    
    #extract the elements from each subset and return the subsets
    jet_0 = x[mask0,:]
    jet_1 = x[mask1,:]
    jet_2_3 = x[mask2_3,:]
    
    #extract the corresponding labels
    label_0 = np.asarray([])
    label_1 = np.asarray([])
    label_2_3 =  np.asarray([])
    if(len(y)>0):
        label_0 = y[mask0]
        label_1 = y[mask1]
        label_2_3 =  y[mask2_3]
    return jet_0, label_0, jet_1, label_1, jet_2_3, label_2_3


def remove_invalid_features(jet_0,jet_1,jet_2_3):
    """For each one of the three sets, filter out the columns (features) that have only invalid (-999) values"""
    #we create a mask of the columns that are invalid for each subset
    invalid_jet_1 = [4,5,6,12,22,23,24,25,26,27,28,29]
    invalid_jet_2 = [4,5,6,12,22,26,27,28]
    
    #we remove the invalid elements from each subset
    jet_0 = np.delete(jet_0,invalid_jet_1,axis=1)
    jet_1 = np.delete(jet_1,invalid_jet_2,axis=1)

    return jet_0,jet_1,jet_2_3
  
    

def remove_outliers(x):
    """Go through every column and check if there are still invalid values (-999). If there are any, replace them by the median of the column"""
    #go through every column and calculate it's mean
    nbColumns = x.shape[1]
    for i in range(nbColumns):
        #we calculate the median of the current column after discarding the -999 values (they should not be in the median)
        median = np.median(x[:,i][x[:,i]!= -999])
        
        #we find the indices of the elements with value -999 in our current column
        indices = x[:,i] == -999
        
        #we replace the element at the found indices by the median of the current column
        x[:,i][indices] = median
    return x

def standardize(x):
    """Standardize the dataset i.e. substract the mean and divide by the standard deviation"""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x

   

def pre_process_data_pipeline(tX,y = np.asarray([])):
    """Pipeline for Data Processing (returns three processed datasets according to their PRI_jet_num and their corresponding labels). This methods basically puts together all the processing steps and makes it easier to call them all together"""

    #group by jet_num
    jet_0, label_0, jet_1, label_1, jet_2_3, label_2_3 = groupy_by_jet_num(tX,y)
    #remove invalid features
    jet_0,jet_1,jet_2_3 = remove_invalid_features(jet_0,jet_1,jet_2_3)    
    #correct reamaining invalid values
    jet_0 = remove_outliers(jet_0)
    jet_1 = remove_outliers(jet_1)    
    jet_2_3 = remove_outliers(jet_2_3)
    #standardize each one of the subsets
    jet_0,_,_ = standardize(jet_0)
    jet_1,_,_ = standardize(jet_1)
    jet_2_3,_,_ = standardize(jet_2_3)
    
    return jet_0, label_0, jet_1, label_1, jet_2_3, label_2_3