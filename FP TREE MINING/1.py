class FPNode:
    def __init__(self, name, count, parent):
        self.name = name
        self.count = count
        self.parent = parent
        self.children = {}
        self.next = None

    def increment(self, count):
        self.count += count

class FPTree:
    def __init__(self, transactions, min_support):
        self.frequent_items = self.find_frequent_items(transactions, min_support)
        self.headers = {}
        self.root = self.build_fptree(transactions, self.frequent_items)

    def find_frequent_items(self, transactions, min_support):
        item_counts = {}
        for transaction in transactions:
            for item in transaction:
                item_counts[item] = item_counts.get(item, 0) + 1
        return {item: count for item, count in item_counts.items() if count >= min_support}

    def build_fptree(self, transactions, frequent_items):
        root = FPNode(None, 1, None)
        for transaction in transactions:
            sorted_items = [item for item in transaction if item in frequent_items]
            sorted_items.sort(key=lambda item: frequent_items[item], reverse=True)
            self.insert_tree(sorted_items, root)
        return root

    def insert_tree(self, items, node):
        if not items:
            return

        first_item = items[0]
        if first_item in node.children:
            node.children[first_item].increment(1)
        else:
            new_node = FPNode(first_item, 1, node)
            node.children[first_item] = new_node
            if first_item in self.headers:
                current = self.headers[first_item]
                while current.next is not None:
                    current = current.next
                current.next = new_node
            else:
                self.headers[first_item] = new_node

        remaining_items = items[1:]
        self.insert_tree(remaining_items, node.children[first_item])

    def mine_patterns(self, min_support):
        patterns = {}
        for item in sorted(self.frequent_items, key=self.frequent_items.get):
            conditional_tree_input = []
            node = self.headers[item]
            while node is not None:
                path = []
                parent = node.parent
                while parent and parent.name:
                    path.append(parent.name)
                    parent = parent.parent
                path.reverse()
                if path:
                    conditional_tree_input.extend([path] * node.count)
                node = node.next
            conditional_tree = FPTree(conditional_tree_input, min_support)
            conditional_patterns = conditional_tree.mine_patterns(min_support)
            if conditional_patterns:
                for pattern, count in conditional_patterns.items():
                    patterns[tuple([item] + list(pattern))] = count
            patterns[(item,)] = self.frequent_items[item]
        return patterns

# Example usage
transactions = [
    ['A', 'B', 'C'],
    ['B', 'C', 'D'],
    ['A', 'C', 'D', 'E'],
    ['A', 'D', 'E'],
    ['A', 'B', 'C', 'E'],
    ['B', 'C']
]

min_support = 2
fp_tree = FPTree(transactions, min_support)
patterns = fp_tree.mine_patterns(min_support)
print("Frequent Patterns:", patterns)