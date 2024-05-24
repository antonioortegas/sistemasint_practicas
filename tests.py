import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight

""" columns for convinience:
index
modifedZurichClass
largestSpotSize
spotDistribution
activity
evolution
previousDailyActivity
historicallyComplex
becameHistoricallyComplex
area
largestSpotArea
cFlares
mFlares
xFlares

1. Code for class (modified Zurich class)  (A,B,C,D,E,F,H)
2. Code for largest spot size              (X,R,S,A,H,K)
3. Code for spot distribution              (X,O,I,C)
4. Activity                                (1 = reduced, 2 = unchanged)
5. Evolution                               (1 = decay, 2 = no growth, 3 = growth)
6. Previous 24 hour flare activity code    (1 = nothing as big as an M1, 2 = one M1, 3 = more activity than one M1)
7. Historically-complex                    (1 = Yes, 2 = No)
8. Did region become historically complex  (1 = yes, 2 = no) on this pass across the sun's disk
9. Area                                    (1 = small, 2 = large)
10. Area of the largest spot               (1 = <=5, 2 = >5)

From all these predictors three classes of flares are predicted, which are represented in the last three columns.

11. C-class flares production by this region    Number in the following 24 hours (common flares)
12. M-class flares production by this region    Number in the following 24 hours (moderate flares)
13. X-class flares production by this region    Number in the following 24 hours (severe flares)
"""

pathToCsv = "./scikit/data/flares.csv"

# load the dataframe from the csv file
df = pd.read_csv(pathToCsv, index_col=0)
# calculate the total number of flares, then write 0 if there are no flares or 1 if there are flares
# and add it as a new column
#df["totalFlares"] = df["cFlares"] + df["mFlares"] + df["xFlares"]
#df["totalFlares"] = df["totalFlares"].apply(lambda x: 1 if x > 0 else 0)

# calculate a new column. the value will be 3 if there are xFlares, 2 if there are mFlares, 1 if there are cFlares, 0 if there are no flares
#TODO
# Calculate a new column 'flareType' based on the priority of flare classes
df['flareType'] = 0  # Initialize flareType column

# Assign values based on priority: X-flares > M-flares > C-flares
df.loc[df['xFlares'] > 0, 'flareType'] = 3  # X-flares
df.loc[(df['mFlares'] > 0) & (df['flareType'] != 3), 'flareType'] = 2  # M-flares
df.loc[(df['cFlares'] > 0) & (df['flareType'] != 2) & (df['flareType'] != 3), 'flareType'] = 1  # C-flares

# eliminate the three columns with the individual flare counts
df = df.drop(columns=["cFlares", "mFlares", "xFlares"])
print(df[(df["flareType"] > 0)].head())
# my df now has 11 columns, 10 of which are features and 1 is the target (last column)

# transform the categorical columns to numerical ones
le = LabelEncoder()
for column in ["modifedZurichClass", "largestSpotSize", "spotDistribution"]:
    df[column] = le.fit_transform(df[column])

X = df.drop(columns=["flareType"])
y = df["flareType"]

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)
"""
"""
# train a model and calculate the accuracy
decision_tree = DecisionTreeClassifier()

# define 3 values for each hyperparameter as a starting point
param_grid = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [5, 10, 15, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

# validate with 10K-fold cross validation
grid_search = GridSearchCV(estimator=decision_tree, param_grid=param_grid, cv=10, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
best_decision_tree = grid_search.best_estimator_
#print(grid_search.cv_results_)
print(best_decision_tree.score(X_test, y_test))


# train a model and calculate the accuracy
multilayer_perceptron = MLPClassifier()

param_grid = {
    'activation': ['logistic', 'tanh', 'relu'],
    'solver': ['adam'],
    'learning_rate': ['adaptive'],
    'max_iter': [50],
    'learning_rate_init': [0.001],
    'verbose': [False],
    'early_stopping': [True],
}

grid_search = GridSearchCV(estimator=multilayer_perceptron, param_grid=param_grid, cv=10, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
best_decision_tree = grid_search.best_estimator_
# get the f1 score
# Get predictions on the test set using the best estimator
y_pred = best_decision_tree.predict(X_test)
print(y_pred)
# Calculate the F1 score
f1 = f1_score(y_test, y_pred)

print("F1 score:", f1)
