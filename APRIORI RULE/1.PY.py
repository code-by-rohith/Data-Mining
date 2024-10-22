import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Sample transaction dataset
data = {
    'Transaction': [
        'Milk, Bread',
        'Bread, Diaper, Beer, Eggs',
        'Milk, Diaper, Beer, Cola',
        'Bread, Milk, Diaper, Beer',
        'Bread, Milk, Cola'
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Convert transactions into a list of lists
transactions = df['Transaction'].apply(lambda x: x.split(', ')).tolist()

# Create a one-hot encoded DataFrame
# Create a list of all items
all_items = list(set(item for sublist in transactions for item in sublist))

# Initialize a DataFrame with the items as columns
onehot = pd.DataFrame(0, index=range(len(transactions)), columns=all_items)

# Fill the DataFrame
for i, transaction in enumerate(transactions):
    for item in transaction:
        onehot.at[i, item] = 1

# Use the Apriori algorithm to find frequent itemsets
min_support = 0.2  # Adjusted minimum support threshold
frequent_itemsets = apriori(onehot, min_support=min_support, use_colnames=True)

# Generate the association rules
min_confidence = 0.5  # Minimum confidence threshold
if not frequent_itemsets.empty:
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
else:
    rules = pd.DataFrame()  # Create an empty DataFrame if no frequent itemsets found
    

# Display results
print("Frequent Itemsets:")
print(frequent_itemsets)
print("\nAssociation Rules:")
print(rules)