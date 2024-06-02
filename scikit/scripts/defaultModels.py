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

# Separate the target column from the rest of the data
y = df["Target"] # only the target column
X = df.drop(columns=["Target"]) # drop the target column

print(df["Target"].value_counts())

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# train a decision tree model and calculate the accuracy
decision_tree = DecisionTreeClassifier() # default parameters

decision_tree.fit(X_train, y_train)
best_decision_tree = decision_tree
# cross validate the model
scores = cross_val_score(best_decision_tree, X_test, y_test, cv=kf)
print("Decision tree accuracy: ", scores.mean())

# train a neural network model and calculate the accuracy
multilayer_perceptron = MLPClassifier()

multilayer_perceptron.fit(X_train, y_train)
best_neural_network = multilayer_perceptron
# cross validate the model
scores = cross_val_score(best_neural_network, X_test, y_test, cv=kf)
print("Multilayer perceptron accuracy: ", scores.mean())