import numpy as np
from helpers.proj1_helpers import *
from helpers.ridge import *
from helpers.processing import *
from helpers.feature_transformation import *
def run(): 

    # Load data
    y, tX, ids = load_csv_data("../train.csv")
    _, tX_test, ids_test = load_csv_data("../test.csv") 

    print("here")

    #Pre-process data (group,remove outliers, standardize)
    jet_0,label_0,jet_1,label_1,jet_2_3, label_2_3= pre_process_data_pipeline(tX,y)

    #Results that we obtained from cross-validation (hyperparameters for each subset):
    best_degree_0 = 4
    best_degree_1 = 7
    best_degree_2_3 = 5
    best_lambda_0 = 1e-07
    best_lambda_1 = 1e-06
    best_lambda_2_3 = 1e-07

    #Results from cross-validation above can be verified by uncommenting the following code
    #*************UNCOMMENT FOR CROSS-VALIDATION**************
    #[lowest_error_0,best_degree_0,best_lambda_0,rmse_tr,rmse_te] = cross_validation_ridge(5,jet_0,label_0)
    #[lowest_error_1,best_degree_1,best_lambda_1,rmse_tr,rmse_te] = cross_validation_ridge(5,jet_1,label_1)
    #[lowest_error_2_3,best_degree_2_3,best_lambda_2_3,rmse_tr,rmse_te] = cross_validation_ridge(5,jet_2_3,label_2_3)
    #*************UNCOMMENT FOR CROSS-VALIDATION**************

    #Polynomial expansion
    jet_0_extended = poly_expansion(jet_0,best_degree_0)
    jet_1_extended = poly_expansion(jet_1,best_degree_1)
    jet_2_3_extended = poly_expansion(jet_2_3,best_degree_2_3)

    # Get weights for each subset (using ridge regression)
    w_jet_0,_ = ridge(label_0,jet_0_extended,best_lambda_0)
    w_jet_1,_ = ridge(label_1,jet_1_extended,best_lambda_1)
    w_jet_2_3,_ = ridge(label_2_3,jet_2_3_extended,best_lambda_2_3)

    #Apply same transformations that we applied to the training_set to the test_set
    jet_0_test,indices_0,jet_1_test,indices_1,jet_2_3_test,indices_2_3 = split_test_set(tX_test,best_degree_0,best_degree_1,best_degree_2_3)

    #Find labels for each one of the subsets of the test_set
    y_pred_jet_0 = predict_labels(w_jet_0,jet_0_test)
    y_pred_jet_0 = y_pred_jet_0.reshape((len(y_pred_jet_0),1))
    y_pred_jet_1 = predict_labels(w_jet_1,jet_1_test)
    y_pred_jet_1 = y_pred_jet_1.reshape((len(y_pred_jet_1),1))
    y_pred_jet_2_3 = predict_labels(w_jet_2_3,jet_2_3_test)
    y_pred_jet_2_3 = y_pred_jet_2_3.reshape((len(y_pred_jet_2_3),1))

    #Merge all results together into one array
    y_pred = np.zeros((tX_test.shape[0],1))

    indices_0 = indices_0.reshape(-1,)
    indices_1 = indices_1.reshape(-1,)
    indices_2_3 = indices_2_3.reshape(-1,)

    y_pred[indices_0] = y_pred_jet_0
    y_pred[indices_1] = y_pred_jet_1
    y_pred[indices_2_3] = y_pred_jet_2_3

    # Create submission
    create_csv_submission(ids_test, y_pred, "output.csv")

run()    
