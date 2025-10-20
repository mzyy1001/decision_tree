import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


class Node:
    def __init__(self, attribute=None, value=None, left=None, right=None, label=None, leaf=False):
        self.attribute = attribute 
        self.value = value
        self.left = left
        self.right = right
        self.label = label
        self.leaf = leaf

    def is_leaf(self):
        return self.leaf


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=float('inf')):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    def _majority_label(self, y):
        vals, counts = np.unique(y, return_counts=True)
        return vals[np.argmax(counts)]

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        if (np.unique(y).size == 1 or
            len(y) < self.min_samples_split or
            depth >= self.max_depth):
            return Node(label=self._majority_label(y), leaf=True)

        best_attr, best_value, best_ig = self._best_split(X, y)

        if best_attr is None or best_ig <= 0:
            return Node(label=self._majority_label(y), leaf=True)

        left_indices = X[:, best_attr] <= best_value
        right_indices = ~left_indices

        left_node = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_node = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        return Node(attribute=best_attr, value=best_value,
                    left=left_node, right=right_node, leaf=False)

    def _best_split(self, X, y):
        best_ig = -np.inf
        best_attr, best_value = None, None
        _, n_features = X.shape

        for attr in range(n_features):
            thresholds = np.unique(X[:, attr])
            for threshold in thresholds:
                left_indices = X[:, attr] <= threshold
                right_indices = ~left_indices

                if left_indices.sum() == 0 or right_indices.sum() == 0:
                    continue

                ig = self._information_gain(y, y[left_indices], y[right_indices])
                if ig > best_ig:
                    best_ig = ig
                    best_attr = attr
                    best_value = threshold

        return best_attr, best_value, best_ig

    def _entropy(self, y):
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()
        return -np.sum(p * np.log2(p))

    def _information_gain(self, y, left_y, right_y):
        n = len(y)
        return ( self._entropy(y)
                 - (len(left_y)/n) * self._entropy(left_y)
                 - (len(right_y)/n) * self._entropy(right_y) )

    def predict(self, X):
        return np.array([self._predict_sample(sample, self.root) for sample in X])

    def _predict_sample(self, sample, node):
        if node.is_leaf():
            return node.label
        if sample[node.attribute] <= node.value:
            return self._predict_sample(sample, node.left)
        else:
            return self._predict_sample(sample, node.right)

    def visualize(self):
        plt.figure(figsize=(12, 8))
        plt.axis('off')
        plt.show()


def main():
    import os

    data_path = os.path.join("wifi_db", "clean_dataset.txt")
    data = np.loadtxt(data_path)

    X = data[:, :-1]
    y = data[:, -1].astype(int)

    np.random.seed(42)
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    split = int(0.8 * len(X))
    train_idx, test_idx = indices[:split], indices[split:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    tree = DecisionTree(min_samples_split=5, max_depth=10)
    tree.fit(X_train, y_train)

    y_pred_train = tree.predict(X_train)
    y_pred_test = tree.predict(X_test)
    acc_train = (y_pred_train == y_train).mean()
    acc_test = (y_pred_test == y_test).mean()

    print(f"Train accuracy: {acc_train * 100:.2f}%")
    print(f"Test  accuracy: {acc_test * 100:.2f}%")

    # tree.visualize()



if __name__ == "__main__":
    main()
