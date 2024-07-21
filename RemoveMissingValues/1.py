import pandas as pd
import numpy as np

# Generate a large dataset with 10000 rows and 10 columns
np.random.seed(42)
rows = 100
columns = 10
data = np.random.randn(rows, columns)

# Introduce missing values randomly
missing_rate = 0.1
mask = np.random.rand(rows, columns) < missing_rate
data[mask] = np.nan

# Create a DataFrame
df_large = pd.DataFrame(data, columns=[f'col_{i}' for i in range(columns)])

# Save the DataFrame to a CSV file
df_large.to_csv('large_dataset.csv', index=False)

print("Large dataset with missing values saved as 'large_dataset.csv'.")
