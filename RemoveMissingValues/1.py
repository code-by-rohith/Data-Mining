import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def count_missing_values(df):
    missing_counts = df.isnull().sum()
    missing_percentage = (missing_counts / len(df)) * 100
    missing_info = pd.concat([missing_counts, missing_percentage], axis=1)
    missing_info.columns = ['Missing Values', 'Percentage']
    return missing_info[missing_info['Missing Values'] > 0]


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        distances = [np.sqrt(np.sum((x - x_train) ** 2)) for x_train in self.X_train]

        k_indices = np.argsort(distances)[:self.k]

        k_nearest_labels = [self.y_train[i] for i in k_indices]

        most_common = max(set(k_nearest_labels), key=k_nearest_labels.count)
        return most_common


def process_dataset(file_name, target_column, drop_columns):
    data = pd.read_csv(file_name)
    data.columns = data.columns.str.strip()

    missing_counts = count_missing_values(data)

    if not missing_counts.empty:
        print(f"DATASET - {file_name} \nMissing Values:")
        print(missing_counts)
    else:
        print(f"\nDATASET - {file_name} \nNo missing values.")

    X = data.drop(drop_columns, axis=1).values
    y = data[target_column].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = KNN(k=3)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy}")

    return data, accuracy


print("Processing the original dataset:")
original_data, original_accuracy = process_dataset('fruits.csv', 'Fruit Category', ['Fruit Category', 'Fruit Name'])

missing_df = pd.read_csv('fruits_missing.csv')

mean_fruit_weight = missing_df['Fruit Weight'].replace(0, np.nan).mean()
missing_df['Fruit Weight'] = missing_df['Fruit Weight'].replace(0, mean_fruit_weight)

missing_df.to_csv('fruits_missing_replaced.csv', index=False)

print("\nProcessing the dataset with missing values replaced by mean:")
missing_data, missing_accuracy = process_dataset('fruits_missing_replaced.csv', 'Fruit Category',
                                                 ['Fruit Category', 'Fruit Name'])

print("\nOriginal Data:")
print(original_data.head())

print("\nData with Missing Values Replaced by Mean:")
print(missing_data.head())

print("\nAccuracy of the original dataset:", original_accuracy)
print("\nAccuracy of the dataset with missing values replaced by mean:", missing_accuracy)

accuracy_difference = original_accuracy - missing_accuracy
print("\nDifference in Accuracy:", accuracy_difference)