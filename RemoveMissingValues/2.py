import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

df1 = pd.read_csv('large_dataset.csv')

df2 = df1.copy()
np.random.seed(42)
mask = np.random.rand(df2.shape[0], df2.shape[1]) < 0.05
df2[mask] = np.nan
df2_filled = df2.copy()
for column in df2.columns:
    mean_value = df2_filled[column].mean()
    df2_filled[column] = df2_filled[column].fillna(mean_value)
print("NaN values in original dataset (df1):")
print(df1.isnull().sum())
print("NaN values in dataset 2 with missing values (df2):")
print(df2.isnull().sum())
print("NaN values in dataset 2 after filling (df2_filled):")
print(df2_filled.isnull().sum())
assert not df2_filled.isnull().values.any(), "There are still NaN values in the filled dataset (df2_filled)."
df1_filled = df1.copy()
for column in df1.columns:
    mean_value = df1_filled[column].mean()
    df1_filled[column] = df1_filled[column].fillna(mean_value)

mse = mean_squared_error(df1_filled, df2_filled)
accuracy = 1 - mse
print("Original Dataset (Dataset 1) Summary:")
print(df1.describe())
print("\nDataset with Missing Values (Dataset 2) Summary:")
print(df2.describe())
print("\nDataset 2 After Filling Missing Values Summary:")
print(df2_filled.describe())
print(f"\nAccuracy: {accuracy}")
