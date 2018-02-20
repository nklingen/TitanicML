"""
UTF-8

Creates a scatter_matrix of the columns in train.csv.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

# Import data
df = pd.read_csv("src/train.csv")

# Inspect data
scatter_matrix(df, alpha=0.2, figsize=(8, 8), diagonal='kde')
plt.show()
