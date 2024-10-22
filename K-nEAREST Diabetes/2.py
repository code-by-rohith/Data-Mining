import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the data
url = 'vehicle.csv'
data = pd.read_csv(url)

# Prepare the data
features = data.drop('Class', axis=1)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)


# Function to calculate the distance matrix
def distance_matrix(X):
    return np.sqrt(((X[:, np.newaxis] - X) ** 2).sum(axis=2))


# Function to compute total cost of clustering
def compute_cost(X, medoids, labels):
    cost = 0
    for i in range(len(medoids)):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            cost += np.sum(distance_matrix(cluster_points)[:, np.argwhere(medoids == i).flatten()])
    return cost


# Function to assign clusters based on the nearest medoid
def assign_clusters(X, medoids):
    distances = distance_matrix(X)
    cluster_assignment = np.argmin(distances[:, medoids], axis=1)
    return cluster_assignment


# K-Medoids algorithm
def k_medoids(X, num_clusters, max_iter=100):
    np.random.seed(42)  # For reproducibility
    medoids = np.random.choice(X.shape[0], size=num_clusters, replace=False)
    previous_medoids = np.copy(medoids)
    for _ in range(max_iter):
        labels = assign_clusters(X, medoids)
        current_cost = compute_cost(X, medoids, labels)

        # Update medoids
        new_medoids = np.copy(medoids)
        for i in range(num_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                costs = np.sum(distance_matrix(cluster_points), axis=1)
                new_medoid = cluster_points[np.argmin(costs)]
                new_medoids[i] = np.where((X == new_medoid).all(axis=1))[0][0]

        # Check for convergence
        if np.array_equal(previous_medoids, new_medoids):
            break
        previous_medoids = np.copy(new_medoids)

    return medoids, labels


# Run K-Medoids
num_clusters = 4
medoids, labels = k_medoids(features_scaled, num_clusters)

# Assign clusters to the data
data['Cluster'] = labels

print("Medoids indices:")
print(medoids)

print("Number of samples per cluster:")
print(data['Cluster'].value_counts())

# Perform PCA for visualization
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features_scaled)

# Map medoids from original space to PCA space for plotting
medoids_pca = pca.transform(features_scaled[medoids])

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(features_pca[:, 0], features_pca[:, 1], c=labels, cmap='viridis', marker='o', alpha=0.6)
plt.scatter(medoids_pca[:, 0], medoids_pca[:, 1], s=300, c='red', marker='X', label='Medoids')
plt.title('K-Medoid Clustering (PCA-reduced)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster')
plt.legend()
plt.show()
