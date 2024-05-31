import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, roc_auc_score, RocCurveDisplay
from sklearn.model_selection import cross_validate
import numpy as np

pathToCsv = "./scikit/data/data.csv"

# load the dataframe from the csv file
df = pd.read_csv(pathToCsv, sep=';')

kf = KFold(n_splits=10, shuffle=True)

# in the last column, substitute all values "Enrolled" with "Graduate"
# this is so that the target column has only 2 values. Since the dataset does not have any discernible way to determine if a student will continue or not
# I will consider only 2 possibilities: the student will drop out, or not (so binary classification)
# this seems to have had the biggest impact on the accuracy, jumping from ~0.73 to ~0.85 in both models
df["Target"] = df["Target"].replace("Enrolled", "Graduate")

# convert the target column to a numerical value
le = LabelEncoder()
df["Target"] = le.fit_transform(df["Target"])

correlation = df.corr()["Target"]
# compute the absolute value of the correlation and sort by it
correlation = correlation.abs().sort_values(ascending=False)
print(correlation)

# Separate the target column from the rest of the data
y = df["Target"] # only the target column
X = df.drop(columns=["Target"]) # drop the target column

# drop all the columns with correlation less than 0.1
columns_to_drop = correlation[correlation < 0.1].index
X = X.drop(columns=columns_to_drop)
print(X.columns)

# Standardize the features
scaler = StandardScaler() # after testing, the type of scaler doesn't seem to matter much, so I'll use the default one
for col in X.columns:
    # scale and return a df column
    X[col] = scaler.fit_transform(X[[col]])
# big impact on the accuracy on the mlp is the scaling of the features when using all the columns
# since some of the eliminated columns have bigger values by orders of magnitude in some cases
# comment the following line and the drop columns one to see the difference in accuracy

# Define bins for discretization (you can adjust the number of bins as needed)
num_bins = 5

# Get the names of the continuous columns
continuous_columns = X.columns

# Discretize continuous columns
for col in continuous_columns:
    X[col] = pd.cut(X[col], bins=num_bins, labels=False)
    
print(X[:30])

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# train a decision tree model and calculate the accuracy
decision_tree = DecisionTreeClassifier()

# define some values for each hyperparameter as a starting point
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [4, 2, 6, 8],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5, 8],
}

# validate with 10K-fold cross validation
grid_search = GridSearchCV(estimator=decision_tree, param_grid=param_grid, cv=kf, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)
best_decision_tree = grid_search.best_estimator_
print("DT: ",best_decision_tree.score(X_test, y_test))
# show each set of hyperparameters and the corresponding accuracy formatted as a string
# results = pd.DataFrame(grid_search.cv_results_)
# results = results[['params', 'mean_test_score']]
# print(results.to_string())

# train a neural network model and calculate the accuracy
multilayer_perceptron = MLPClassifier()

# define some values for each hyperparameter as a starting point
param_grid = {
    'hidden_layer_sizes': [(50,),(20,),(20, 20, 20)],
    'solver': ['adam'],
    'max_iter': [500],
    'alpha': [0.02],
    'learning_rate_init': [0.01, 0.1],
    'learning_rate': ['adaptive'],
    'activation': ['relu'],
    'early_stopping': [True],
}
# validate with 10K-fold cross validation
grid_search = GridSearchCV(estimator=multilayer_perceptron, param_grid=param_grid, cv=kf, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)
best_neural_network = grid_search.best_estimator_
print("MLP: ", best_neural_network.score(X_test, y_test))
# show each set of hyperparameters and the corresponding accuracy formatted as a string
# results = pd.DataFrame(grid_search.cv_results_)
# results = results[['params', 'mean_test_score']]
# print(results.to_string())

# use cross_validate to get some more metrics
# metrics to calculate
scoring = ['f1', 'precision', 'recall']

# Evaluate Decision Tree
decision_tree_scores = cross_validate(best_decision_tree, X_test, y_test, cv=kf, scoring=scoring)
# print(decision_tree_scores)
print("Decision Tree")
for metric in scoring:
    print(f"{metric}: {decision_tree_scores['test_' + metric].mean()}")

# Evaluate Neural Network
multilayer_perceptron_scores = cross_validate(best_neural_network, X_test, y_test, cv=kf, scoring=scoring)
print("Neural Network")
# cross_validate returns a dictionary with the scores for each metric in an array
# to access each score, we need to access the key 'test_' + name_of_the_metric
# then we can calculate the mean for that metric
for metric in scoring:
    print(f"{metric}: {multilayer_perceptron_scores['test_' + metric].mean()}")
    
# TODO ROC curves