import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('fruit_data_with_colours.csv')

data = df[['mass', 'width']].values


def k_means_clustering(data, k, max_iterations=100):
    centroids = data[np.random.choice(len(data), k, replace=False)]

    def euclidean_distance(x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def assign_clusters(data, centroids):
        clusters = [[] for _ in range(len(centroids))]

        for point in data:
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            cluster_idx = np.argmin(distances)
            clusters[cluster_idx].append(point)

        return clusters

    def update_centroids(clusters):
        centroids = np.zeros((len(clusters), clusters[0][0].shape[0]))

        for i, cluster in enumerate(clusters):
            centroids[i] = np.mean(cluster, axis=0)

        return centroids

    for _ in range(max_iterations):
        clusters = assign_clusters(data, centroids)
        new_centroids = update_centroids(clusters)

        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return centroids, clusters


k = 3

centroids, clusters = k_means_clustering(data, k)

colors = ['r', 'g', 'b']
for i, cluster in enumerate(clusters):
    cluster = np.array(cluster)
    plt.scatter(cluster[:, 0], cluster[:, 1], c=colors[i], label=f'Cluster {i + 1}')

plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=100, c='black', label='Centroids')
plt.title('K-means Clustering')
plt.xlabel('Mass')
plt.ylabel('Width')
plt.legend()
plt.show()