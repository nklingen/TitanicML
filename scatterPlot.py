import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

# Import data
df = pd.read_csv("train.csv")

# Inspect data
scatter_matrix(df, alpha=0.2, figsize=(12, 12), diagonal='kde')
plt.show()
