"""
UTF-8

Applies machine learning algorithms to train.csv and
separates the data into training and testing data.

Simon's token: c6e56634-1e08-49dc-8c74-eff8fe36bc3c
"""

import os

import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# os.chdir(os.path.dirname(__file__))

# Import data
df = pd.read_csv("src/train.csv")

# Assign features
features = ["Embarked", "Pclass", "Sex", "Age", "Parch", "SibSp", "Cabin"]

# Convert Sex to binary, integers.
# Male = 0, Female = 1
df = df.replace(to_replace="male", value=0)
df = df.replace(to_replace="female", value=1)

# Convert embarked to C = 0, S = 1, Q = 2
df = df.replace(to_replace="C", value=0)
df = df.replace(to_replace="S", value=1)
df = df.replace(to_replace="Q", value=2)

# Convert cabin to binary, so it tells us whether or not they had a cabin
# Set value to 0 if NaN, else 1
df["Cabin"] = df["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

# For Nat:
# .apply method to the Series instance takes a function as the first argument
# that's why we use the lambda function (a function without a name)
# which basically iterates over the Series as in:
# for elem in [1, 2, "str", 4]:
#     if type(elem) == str:
#         print("Omg mad stringz")
#
# Hope it makes sense, text me if it doesn't

# Remove corrupt data
df = df[features + ["Survived"]].dropna()

# Split data into test and training data
df_train = df.iloc[0:int(len(df) * 0.75), :]
df_test = df.iloc[int(len(df) * 0.75): len(df), :]

# Choose features as sex and age
X_test = df_test[features]
X_train = df_train[features]

# Choose target as whether or not PassengerId survived
y_train = df_train["Survived"]
y_test = df_test["Survived"]

# Create data for Kfolds method
X_kfold = df[features]
y_kfold = df["Survived"]


# Create classifier instances
svm_holdout = SVC(C=1.5, kernel="rbf")
knn_holdout = KNeighborsClassifier(n_neighbors=5)
dt_holdout = DecisionTreeClassifier()

svm_kfold = SVC(C=1.5, kernel="linear")
knn_kfold = KNeighborsClassifier(n_neighbors=5)
dt_kfold = DecisionTreeClassifier()

# Fit data to classifiers
svm_holdout.fit(X_train, y_train)
knn_holdout.fit(X_train, y_train)
dt_holdout.fit(X_train, y_train)

# Score models to test data, using the Holdout method
svm_holdout_scores = svm_holdout.score(X_test, y_test)
knn_holdout_scores = knn_holdout.score(X_test, y_test)
dt_holdout_scores = dt_holdout.score(X_test, y_test)

# Score models to using the KFold method
svm_kfold_scores = cross_val_score(svm_kfold, X_kfold, y_kfold, cv=10)
knn_kfold_scores = cross_val_score(knn_kfold, X_kfold, y_kfold, cv=10)
dt_kfold_scores = cross_val_score(dt_kfold, X_kfold, y_kfold, cv=10)


df_kfold_scores = pd.DataFrame(data={"SVM Scores": svm_kfold_scores,
                                     "KNN Scores": knn_kfold_scores,
                                     "DT Scores": dt_kfold_scores})

df_holdout_scores = pd.DataFrame(data={"SVM Scores": svm_holdout_scores,
                                       "KNN Scores": knn_holdout_scores,
                                       "DT Scores": dt_holdout_scores},
                                 index=["Accuracy"])

print("\n\nHoldout Scores")
print(df_holdout_scores)
print("\n\nKfold Scores")
print(df_kfold_scores)

# Might be possible the model is overfitted a bit, we should look into
# how to properly evaluate and fit the models with correct features
# Also the Decision Tree needs less categories to decide from
# Meaning all under age 16, gets value 0. Those between 16 and 25 gets
# the value 1 and so on
