import pandas as pd
import numpy as np

# Parameters
num_rows = 1000
num_features = 13

# Generate random data
np.random.seed(42)  # For reproducibility
data = np.random.randint(0, 2, size=(num_rows, num_features))
classes = np.random.randint(1, 5, size=num_rows)  # Random classes from 1 to 4

# Create DataFrame
df = pd.DataFrame(data, columns=[str(i) for i in range(1, num_features + 1)])
df.insert(0, 'Class', classes)  # Insert 'Class' column at the beginning

# Save to CSV
df.to_csv('vehicle.csv', index=False)
