import csv
import random

def generate_random_data(num_points, num_features):
    return [[random.uniform(0, 10) for _ in range(num_features)] for _ in range(num_points)]

def save_data_to_csv(file_path, data):
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([f'Feature{i+1}' for i in range(len(data[0]))])
        writer.writerows(data)

if __name__ == "__main__":
    num_points = 100
    num_features = 2
    file_path = 'kmeansdataset.csv'
    data = generate_random_data(num_points, num_features)
    save_data_to_csv(file_path, data)
    print(f"Generated dataset with {num_points} points and {num_features} features saved to {file_path}.")
