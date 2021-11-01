# Machine Learning Project 1 - Higgs Boson Dataset

This is our attempt to solve the Higgs Boson challenge for the EPFL Machine Learning CS-433.
Fall 2021

## Result:

Our final score was 81% on CrowdAI for the given test-set.

## Team:

ADEYE Abiola: [abiola.adeye@epfl.ch](mailto:abiola.adeye@epfl.ch)

RAHMOUN Aya: [aya.rahmoun@epfl.ch](mailto:aya.rahmoun@epfl.ch)

GUESSOUS Abdeslam: [abdeslam.guessous@epfl.ch](mailto:abdeslam.guessous@epfl.ch)

## Requirements:

No external libraries were allowed for this project, except numpy and data-visualization libraries

## Data :

Because the training set is larger than 100mb, we could not directly put it in the git repository. You can find the test and trainig sets at the following link:
https://www.kaggle.com/c/higgs-boson/data

## Files

- `implementations.py` : implementation of all the models that we saw
- `costs.py` : contains different cost functions that are needed for training our models
- `cross_validation.py` : contains all the steps needed for cross-validation
- `feature_transformation.py` : contains different methods for transforming our features to improve accuracy
- `processing.py` : contains different functions for pre-processing our data-set
- `gradient_descent.py` : implementation of least squares gradient descent
- `least_squares.py` : implementation of least squares with ols estimator
- `stochastic_gradient_descent.py` : implementation of least squares using stochastic gradient descent
- `logistic_reg.py` : implementation of logistic and regularized logistic regression
- `ridge.py` : implementation of least squares using penalized linear regression
- `proj1_helpers.py` : provides extra functions that are helpful for the project
- `project1.ipynb` : jupyter notebook that contains all the steps from data cleansing until submission
- `Data_Visualization.ipynb` : jupyter notebook were we did all the preliminary data-visualization
- `run.py` : script that reproduces our final submission
