import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Import data
df = pd.read_csv("train.csv")

# Assign features
features = ["Embarked", "Pclass", "Sex", "Age", "Parch", "SibSp"]

# Remove corrupt data
df = df[features + ["Survived"]].dropna()

# Convert Sex to binary, integers.
# Male = 0, Female = 1
df = df.replace(to_replace="male", value=0)
df = df.replace(to_replace="female", value=1)

# Convert embarked to C = 0, S = 1, Q = 2
df = df.replace(to_replace="C", value=0)
df = df.replace(to_replace="S", value=1)
df = df.replace(to_replace="Q", value=2)

# Split data into test and training data
df_train = df.iloc[0:int(len(df) * 0.75), :]
df_test = df.iloc[int(len(df) * 0.75): len(df), :]

# Choose features as sex and age
X_test = df_test[features]
X_train = df_train[features]

# Choose target as whether or not PassengerId survived
y_train = df_train["Survived"]
y_test = df_test["Survived"]

# Create svm classififer, with default parameters
svm = SVC(C=1.5, kernel="rbf")

# Fit data
svm.fit(X_train, y_train)

svm.score(X_test, y_test)

# Create knn classififer
knn = KNeighborsClassifier(n_neighbors=5)

# Fit data
knn.fit(X_train, y_train)

# score
knn.score(X_test, y_test)

# Create dt classifier
dt = DecisionTreeClassifier()

# Fit data
dt.fit(X_train, y_train)

# score
dt.score(X_test, y_test)
