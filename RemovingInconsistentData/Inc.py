import csv
import random
import matplotlib.pyplot as plt


def generate_cluster_data(num_clusters, points_per_cluster, spread=1.0):
    """
    Generates a dataset with distinct clusters.

    Args:
        num_clusters (int): Number of clusters to generate.
        points_per_cluster (int): Number of points per cluster.
        spread (float): Spread of points around each cluster center.

    Returns:
        list of lists: Generated data points.
    """
    data = []
    for i in range(num_clusters):
        center_x = i * 10  # X-coordinate of the cluster center
        center_y = i * 10  # Y-coordinate of the cluster center
        for _ in range(points_per_cluster):
            x = center_x + spread * random.uniform(-1, 1)
            y = center_y + spread * random.uniform(-1, 1)
            data.append([x, y])
    return data


def save_data_to_csv(file_path, data):
    """
    Saves the generated data to a CSV file.

    Args:
        file_path (str): Path to the output CSV file.
        data (list of lists): Data to save.
    """
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Feature1', 'Feature2'])
        writer.writerows(data)


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
    # Generate and save the dataset
    num_clusters = 5
    points_per_cluster = 20
    spread = 1.0
    file_path = 'kmeansdataset_consistent.csv'

    data = generate_cluster_data(num_clusters, points_per_cluster, spread)
    save_data_to_csv(file_path, data)
    print(
        f"Generated dataset with {num_clusters} clusters, each with {points_per_cluster} points, saved to {file_path}.")

    # Load the dataset
    data = load_data(file_path)

    # Apply K-means clustering
    k = num_clusters
    clusters, centroids = k_means(data, k)

    # Print final centroids and clusters
    print(f"Final centroids: {centroids}")
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i}: {cluster}")

    # Plot the clusters and centroids
    plot_clusters(clusters, centroids)
