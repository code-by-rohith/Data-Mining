import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances


def k_medoids(X, k, max_iter=100):
    m, n = X.shape
    medoids = np.random.choice(m, k, replace=False)
    clusters = np.zeros(m)

    for _ in range(max_iter):
        distances = pairwise_distances(X, X[medoids])
        clusters = np.argmin(distances, axis=1)

        new_medoids = np.copy(medoids)
        for i in range(k):
            cluster_points = np.where(clusters == i)[0]
            intra_cluster_distances = pairwise_distances(X[cluster_points], X[cluster_points])
            total_distance = np.sum(intra_cluster_distances, axis=1)
            new_medoids[i] = cluster_points[np.argmin(total_distance)]

        if np.all(medoids == new_medoids):
            break

        medoids = new_medoids

    return medoids, clusters


def plot_k_medoids(X, medoids, clusters):
    plt.figure(figsize=(8, 6))

    # Plot data points
    plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', marker='o', label='Data Points')

    # Plot medoids
    plt.scatter(X[medoids][:, 0], X[medoids][:, 1], c='red', marker='X', s=200, label='Medoids')

    plt.title('K-Medoids Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid()
    plt.show()


X = np.array([[2, 3], [3, 3], [8, 8], [9, 9], [4, 2], [7, 8]])
k = 2

# Run k-medoids
medoids, clusters = k_medoids(X, k)

# Plot the results
plot_k_medoids(X, medoids, clusters)