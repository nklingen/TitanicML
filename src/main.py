"""
UTF-8

Applies machine learning algorithms to train.csv and
separates the data into training and testing data.
"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

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


# Create classifier instances
svm = SVC(C=1.5, kernel="rbf")
knn = KNeighborsClassifier(n_neighbors=5)
dt = DecisionTreeClassifier()

# Fit data to classifiers
svm.fit(X_train, y_train)
knn.fit(X_train, y_train)
dt.fit(X_train, y_train)

# Score models to test data
svm.score(X_test, y_test)
knn.score(X_test, y_test)
dt.score(X_test, y_test)
