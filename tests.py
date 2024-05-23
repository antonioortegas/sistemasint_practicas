import pandas as pd

""" columns for convinience:
index
modifedZurichClass
largestSpotSize
spotDistribution
activitt
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
# compute the total number of flares for each row and add it as a new column
df["totalFlares"] = df["cFlares"] + df["mFlares"] + df["xFlares"]
# eliminate the three columns with the individual flare counts
df = df.drop(columns=["cFlares", "mFlares", "xFlares"])
#print(df[(df["totalFlares"] > 0)].head())
# my df now has 11 columns, 10 of which are features and 1 is the target (last column)

#TODO: maybe transform the categorical columns to numerical ones

X = df.drop(columns=["totalFlares"])
y = df["totalFlares"]
print(X.head())
print(y.head())

#TODO Split the data into training and testing sets
