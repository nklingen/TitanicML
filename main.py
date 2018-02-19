import pandas as pd
from sklearn.svm import SVC

# Import data
df = pd.read_csv("train.csv")

# Assign features
features = ["Sex", "Age", "SibSp", "Parch", "Pclass"]

# Remove corrupt data
df = df[features + ["Survived"]].dropna()


# Convert Sex to binary, integers.
# Male = 0, Female = 1
df = df.replace(to_replace="male", value=0)
df = df.replace(to_replace="female", value=1)

# Split data into test and training data
df_train = df.iloc[0:int(len(df) * 0.75), :]
df_test = df.iloc[int(len(df) * 0.75): len(df), :]

# Choose features as sex and age
X_test = df_test[features]
X_train = df_train[features]

# Choose target as whether or not PassengerId survived
y_train = df_train["Survived"]
y_test = df_test["Survived"]

# Create classififer, with default parameters

svm = SVC()

# Fit data
svm.fit(X_train, y_train)

svm.score(X_test, y_test)
