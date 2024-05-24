import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

pathToCsv = "./scikit/data/data.csv"
kf = KFold(n_splits=10)

# load the dataframe from the csv file
# the csv has ; as separator instead of , for some reason
df = pd.read_csv(pathToCsv, sep=';')
# in the last column, substitute all values "Enrolled" with "Graduate"
# this is so that the target column has only 2 values. Since the dataset does not have any discernible way to determine if a student will continue or not
# I will consider only 2 possibilities: the student will drop out, or not (so binary classification)
# this seems to have had the biggest impact on the accuracy, jumping from ~0.73 to ~0.85 in both models
df["Target"] = df["Target"].replace("Enrolled", "Graduate")
print(df.head())

# convert the target column to a numerical value, 0 for Dropout, 1 for Enrolled, 2 for Graduate
le = LabelEncoder()
df["Target"] = le.fit_transform(df["Target"])
#print(df.head())

X = df.drop(columns=["Target"]) # drop the target column
# drop all the columns that dont seem to have any impact on the accuracy (tested by removing them one by one and checking the accuracy)
X = X.drop(X.columns[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,]], axis=1)
print(X.columns)
# column 16 (tuition fees up to date) seems to have a big impact on the accuracy (about 0.05). Seems plausible. If the student can't pay the fees, they will have to drop out
# columns 29 to 32 (related to the performance of the student on the 2nd semester) seem to have a 0.2-0.3 impact on the accuracy
# my guess is that if the student is doing well on the 2nd semester, they are more likely to continue. Also, they most likely did well on the 1st semester too, so the 1st semester columns are not needed
# columns 33 to 35 (related to the economic situation [inflation, unemployment, ans GDP]) seem to have a 0.2-0.3 impact on the accuracy
# the accuracy seems to stabilize at around 0.75 before removing any columns, and removing any of the columns mentioned above seems to keep it there

# Standardize the features
scaler = StandardScaler() # after testing, the type of scaler doesn't seem to matter much, so I'll use the default one
X_scaled = scaler.fit_transform(X)
# big impact on the accuracy on the mlp is the scaling of the features when using all the columns
# since some of the eliminated columns have bigger values by orders of magnitude in some cases
# comment the following line and the drop columns one to see the difference in accuracy
X = X_scaled

y = df["Target"]


# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# train a decision tree model and calculate the accuracy
decision_tree = DecisionTreeClassifier()

# define some values for each hyperparameter as a starting point
param_grid = {
    'criterion': ['gini'], # doesn't seem to matter much, so I'll fix it to gini
    'max_depth': [2, 3, 4, 5, 7, ], # 10 is already overfitting, so i will try around 5. 3 to 4 seems to be the best
    'min_samples_split': [2, 4], # doesn't seem to matter much, so I'll fix it to 2 which is the default
    'min_samples_leaf': [1, 2], # doesn't seem to matter much, so I'll fix it to 1 which is the default
    # overall this seems to improve the accuracy from 0.79 to 0.85
}

# validate with 10K-fold cross validation
grid_search = GridSearchCV(estimator=decision_tree, param_grid=param_grid, cv=kf, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)
best_decision_tree = grid_search.best_estimator_
print(best_decision_tree.score(X_test, y_test))
# show each set of hyperparameters and the corresponding accuracy formatted as a string
results = pd.DataFrame(grid_search.cv_results_)
results = results[['params', 'mean_test_score']]
print(results.to_string())

# train a neural network model and calculate the accuracy
multilayer_perceptron = MLPClassifier()

# define some values for each hyperparameter as a starting point
param_grid = {
    'hidden_layer_sizes': [(100,), (100, 100), (10,), (20, 20,)], # doesn't seem to matter much
    'solver': ['adam'], # doesn't seem to matter much
    'max_iter': [500],
    'alpha': [0.1,0.2, 0.5, 0.01], # between 0.1 and 0.5 seems to be ok
    'learning_rate_init': [0.01],
    'activation': ['relu'], # doesn't seem to matter much
}

grid_search = GridSearchCV(estimator=multilayer_perceptron, param_grid=param_grid, cv=kf, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)
best_neural_network = grid_search.best_estimator_
print(best_neural_network.score(X_test, y_test))
# show each set of hyperparameters and the corresponding accuracy formatted as a string
results = pd.DataFrame(grid_search.cv_results_)
results = results[['params', 'mean_test_score']]
print(results.to_string())

# TODO plotting the ROC curve for both best models
# TODO show various metrics for comparison