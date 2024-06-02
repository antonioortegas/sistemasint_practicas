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

# replace “Graduate” with “Enrolled”
df["Target"] = df["Target"].replace("Enrolled", "Graduate")

print(df["Target"].value_counts())

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
# after testing, the TYPE of scaler doesn't seem to matter much
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = X_scaled

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# train a decision tree model and calculate the f1_score
decision_tree = DecisionTreeClassifier() # default parameters

# define some values for each hyperparameter as a starting point
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [4, 2, 6, 8],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5, 8],
}

# validate with 10K-fold cross validation
grid_search = GridSearchCV(estimator=decision_tree, param_grid=param_grid, cv=kf, scoring='f1', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)
best_decision_tree = grid_search.best_estimator_
# cross validate the model
scores = cross_val_score(best_decision_tree, X_test, y_test, cv=kf, scoring='f1')
print("Decision tree f1_score: ", scores.mean())

# train a neural network model and calculate the f1_score
multilayer_perceptron = MLPClassifier()

# define some values for each hyperparameter as a starting point
param_grid = {
    'hidden_layer_sizes': [(20, 20)],
    'solver': ['adam'],
    'max_iter': [500],
    'learning_rate': ['adaptive'],
    'activation': ['relu'],
    'early_stopping': [True],
}
grid_search = GridSearchCV(estimator=multilayer_perceptron, param_grid=param_grid, cv=kf, scoring='f1', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)
best_neural_network = grid_search.best_estimator_
# cross validate the model
scores = cross_val_score(best_neural_network, X_test, y_test, cv=kf, scoring='f1')
print("Multilayer perceptron f1_score: ", scores.mean())

# use cross_validate to get some more metrics
# metrics to calculate
scoring = ['accuracy', 'precision', 'recall']

# Evaluate Decision Tree
decision_tree_scores = cross_validate(best_decision_tree, X_test, y_test, cv=kf, scoring=scoring)
# cross_validate returns a dictionary with the scores for each metric in an array
# to access each score, we need to access the key 'test_' + name_of_the_metric
# then we can calculate the mean for that metric
print("Decision Tree")
for metric in scoring:
    print(f"{metric}: {decision_tree_scores['test_' + metric].mean()}")

# Evaluate Neural Network
multilayer_perceptron_scores = cross_validate(best_neural_network, X_test, y_test, cv=kf, scoring=scoring)
print("Neural Network")
for metric in scoring:
    print(f"{metric}: {multilayer_perceptron_scores['test_' + metric].mean()}")
    
# Using .from_estimator() to generate ROC curve
# ROC curve for Decision Tree
roc_disp = RocCurveDisplay.from_estimator(best_decision_tree, X_test, y_test)
plot = roc_disp.plot()
plt.show()

# ROC curve for Neural Network
roc_disp = RocCurveDisplay.from_estimator(best_neural_network, X_test, y_test)
plot = roc_disp.plot()
plt.show()

