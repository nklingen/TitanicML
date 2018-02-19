import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

# Import data
df = pd.read_csv("train.csv")

# Inspect data
scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='kde')
plt.show()

# Remove corrupt data
df = df[["PassengerId", "Sex", "Age", "Survived"]].dropna()

# Convert Sex to binary, integers.
# Male = 0, Female = 1
df = df.replace(to_replace="male", value=0)
df = df.replace(to_replace="female", value=1)

# Split data into test and training data
df_train = df.iloc[0:int(len(df) * 0.75), :]
df_test = df.iloc[int(len(df) * 0.75): len(df), :]

# Choose features as sex and age
X_train = df_train[["PassengerId", "Sex", "Age"]]
X_test = df_test[["PassengerId", "Sex", "Age"]]

# Choose target as whether or not PassengerId survived
y_train = df_train["Survived"]
y_test = df_test["Survived"]

# Create classififer, with default parameters
svm = SVC()

# Fit data
svm.fit(X_train, y_train)

svm.score(X_test, y_test)

plt.scatter(df["Sex"], df["Age"], c=df["Survived"])
plt.show()
