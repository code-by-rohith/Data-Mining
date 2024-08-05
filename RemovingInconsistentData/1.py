import pandas as pd

file_path = "diabetes.csv"
df = pd.read_csv(file_path)

print("Initial data:")
print(df.head())

print("\nDuplicate rows before cleaning:")
duplicates = df[df.duplicated()]
print(duplicates)

print("\nNumber of duplicate rows before cleaning:")
print(df.duplicated().sum())

df.drop_duplicates(inplace=True)

print("\nNumber of duplicate rows after cleaning:")
print(df.duplicated().sum())

df['Outcome'] = df['Outcome'].astype('category')

print("\nCleaned data:")
print(df.head())

df.to_csv("cleaned_diabetes.csv", index=False)
print("\nCleaned dataset saved to 'cleaned_diabetes.csv'.")