from sklearn.ensemble import IsolationForest
import numpy as np

# Generate sample data
rng = np.random.RandomState(42)
# Generate 100 normal data points
X_normal = 0.3 * rng.randn(10, 2)
# Generate 20 anomalies
X_anomalies = rng.uniform(low=-4, high=4, size=(20, 2))
# Combine normal data and anomalies
X = np.r_[X_normal + 2, X_anomalies]

# Initialize Isolation Forest model
model = IsolationForest(contamination=0.2, random_state=rng)
# Fit the model
model.fit(X)
# Predict anomalies (-1 indicates an anomaly, 1 indicates normal data)
predictions = model.predict(X)

# Output results
for i, prediction in enumerate(predictions):
    if prediction == -1:
        print(f"Data point {i} is an anomaly: {X[i]}")
    else:
        print(f"Data point {i} is normal: {X[i]}")