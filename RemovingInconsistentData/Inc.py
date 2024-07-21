import csv
import random
import matplotlib.pyplot as plt

def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        for row in reader:
            data.append([float(row[0]), float(row[1])])
    return data

def initialize_centroids(data, k):
    return random.sample(data, k)

def euclidean_distance(point1, point2):
    return sum((x - y) ** 2 for x, y in zip(point1, point2)) ** 0.5

def assign_clusters(data, centroids):
    clusters = [[] for _ in centroids]
    for point in data:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        closest_centroid_index = distances.index(min(distances))
        clusters[closest_centroid_index].append(point)
    return clusters

def recalculate_centroids(clusters):
    new_centroids = []
    for cluster in clusters:
        if len(cluster) == 0:
            continue
        new_centroid = [sum(dim) / len(cluster) for dim in zip(*cluster)]
        new_centroids.append(new_centroid)
    return new_centroids

def has_converged(old_centroids, new_centroids, threshold=1e-4):
    total_movement = sum(euclidean_distance(old, new) for old, new in zip(old_centroids, new_centroids))
    return total_movement < threshold

def k_means(data, k, max_iterations=100):
    centroids = initialize_centroids(data, k)
    for _ in range(max_iterations):
        clusters = assign_clusters(data, centroids)
        new_centroids = recalculate_centroids(clusters)
        if has_converged(centroids, new_centroids):
            break
        centroids = new_centroids
    return clusters, centroids

def plot_clusters(clusters, centroids):
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    for i, cluster in enumerate(clusters):
        if len(cluster) == 0:
            continue
        cluster_points = list(zip(*cluster))
        plt.scatter(cluster_points[0], cluster_points[1], c=colors[i % len(colors)], label=f'Cluster {i}')
    if centroids:
        centroid_points = list(zip(*centroids))
        plt.scatter(centroid_points[0], centroid_points[1], c='k', marker='x', label='Centroids')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.title('K-Means Clustering')
    plt.show()

if __name__ == "__main__":
    # Update the file path if necessary
    data = load_data('kmeansdataset.csv')
    k = 5
    clusters, centroids = k_means(data, k)
    print(f"Final centroids: {centroids}")
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i}: {cluster}")
    plot_clusters(clusters, centroids)
