import numpy as np


# Function to split the dataset into Training and Test Set
def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    splits the dataset into train and test sets, stratified by the class label.

    Parameters:
    X: predictors (input data).
    y: target variable (class labels).
    test_size: 20% of the data is test set
    random_state (int): Random seed for reproducibility

    Returns:
    X_train, X_test, y_train, y_test
    """
    # set random seed for reproducibility
    np.random.seed(random_state)

    # number of samples
    n_samples = len(X)

    # shuffle the indices
    shuffled_indices = np.random.permutation(np.arange(n_samples))

    # determine the test set size
    test_size = int(n_samples * test_size)

    # split the indices into train and test
    train_indices = shuffled_indices[test_size:]
    test_indices = shuffled_indices[:test_size]

    # split the features and target arrays into test and train
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test

# Node class
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, is_leaf=False, prediction=None):
        """
        initialize a node for the decision tree
        :param feature: the feature of the dataset
        :param threshold: the condition value used to split the feature
        :param left: the left child nodes which don't meet the condition
        :param right: the right child nodes which meet the condition
        :param is_leaf: set default to false (not a terminal node)
        :param prediction: class label if the node is terminal
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        self.prediction = prediction

    def decide(self, x):
        """
        decision function determining which branch to follow given input x
        :param x: an instance (data point) from the Training set or Test set (a vector, or a numpy array)
        :return: True for right branch and False for left branch (Boolean)
        """
        if self.is_leaf:
            return self.prediction

        value = x[self.feature]  # Get the correct feature value from x
        if value >= self.threshold:
            return self.right.decide(x)  # Call `decide()` recursively
        else:
            return self.left.decide(x)


# DecisionTreeClassifier class
class DecisionTreeClassifier:
    """
    a decision tree classifier for binary classification
    """

    def __init__(self, min_samples=2, max_depths=None, criterion='entropy'):
        """
        initialize the decision tree class
        :param min_samples: the minimum number of samples in a node allowing the tree to split
        :param max_depths: the maximum levels controlling how deep the tree can grow
        """
        self.tree = None
        self.min_samples = min_samples
        self.max_depths = max_depths
        self.criterion = criterion

    def get_params(self, deep=True):
        return {
            'max_depth': self.max_depths,
            'min_samples': self.min_samples,
            'criterion': self.criterion
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

    def _entropy(self, y):
        """
        Entropy measures impurity in a dataset (when classes are evenly mixed)
        :param y: the label values (numpy array)
        :return: the entropy value (float64)
        """
        entropy = 0  # pure set

        # find the unique label values in y and count them
        labels, counts = np.unique(y, return_counts=True)

        total_samples = len(y)

        # compute the probability of each class, and the entropy value
        # loop through each class count
        for count in counts:
            p = count / total_samples
            entropy += -p * np.log2(p + 1e-9)  # to avoid log(0)
        return entropy

    def _information_gain(self, y, y_left, y_right):
        """
        Information Gain measures how much uncertainty is reduced after splitting a dataset
        :param y: the label class before splitting
        :param y_left: the labels in the left branch after splitting
        :param y_right: tbe labels in the right branch after splitting
        :return: information gain of the split (the higher the gain the better it is, which means entropy is low)
        """
        # compute entropy before splitting
        entropy_before = self._entropy(y)

        # calculate weights (probability of samples in each branch)
        weight_left = len(y_left) / len(y)
        weight_right = len(y_right) / len(y)

        # compute the entropy for left and right branches
        entropy_left = self._entropy(y_left)
        entropy_right = self._entropy(y_right)

        # compute entropy after the split
        entropy_after = weight_left * entropy_left + weight_right * entropy_right

        # compute Information Gain (entropy before - entropy after)
        return entropy_before - entropy_after

    def _gini(self, y):
        """
        gini determines how pure each split is (closer to 0 mean purity)
        :param y: the label values before split
        :return: gini value (the lower the value, the better split it is)
        """
        # get unique values for the label, and count them
        labels, counts = np.unique(y, return_counts=True)
        total_samples = len(y)

        # compute the probability of each class
        probabilities = counts / total_samples

        # compute the gini impurity
        gini = 1 - np.sum(probabilities ** 2)

        return gini

    def _gini_gain(self, y, y_left, y_right):
        # compute entropy before splitting
        gini_before = self._gini(y)

        # calculate weights (probability of samples in each branch)
        weight_left = len(y_left) / len(y)
        weight_right = len(y_right) / len(y)

        # compute the entropy for left and right branches
        gini_left = self._gini(y_left)
        gini_right = self._gini(y_right)

        # compute entropy after the split
        gini_after = weight_left * gini_left + weight_right * gini_right

        # compute Information Gain (entropy before - entropy after)
        return gini_before - gini_after

    def _chi_square_split(self, y, y_left, y_right):
        """
        a high chi-square value means a stronger relationship between the features and the target
        :param y: the target label (numpy array)
        :param y_left: the labels in the left branch
        :param y_right: the labels in the right branch
        :return: the chi-square value of the split
        """
        unique_classes = np.unique(y)

        def get_class_counts(y_subset):
            class_counts = np.zeros(len(unique_classes))  # Initialize count array
            subset_classes, subset_counts = np.unique(y_subset, return_counts=True)
            for i, cls in enumerate(unique_classes):  # Align counts with full class list
                if cls in subset_classes:
                    class_counts[i] = subset_counts[np.where(subset_classes == cls)[0][0]]
            return class_counts

        # find the unique labels of y, y_left, y_right, and count them
        parent_counts = get_class_counts(y)
        left_counts = get_class_counts(y_left)
        right_counts = get_class_counts(y_right)

        # find the number of samples for y, y_left, y_right
        total_samples = len(y)
        total_left, total_right = len(y_left), len(y_right)

        # find the expected frequency of y_left and y_right
        expected_left = parent_counts * (total_left / total_samples)
        expected_right = parent_counts * (total_right / total_samples)

        # Avoid division by zero
        expected_left = np.where(expected_left == 0, 1e-9, expected_left)
        expected_right = np.where(expected_right == 0, 1e-9, expected_right)

        chi_square = np.sum((left_counts - expected_left) ** 2 / expected_left) + \
                     np.sum((right_counts - expected_right) ** 2 / expected_right)

        return chi_square

    def _find_best_split(self, X, y):
        """
        find the best split (the best feature and threshold) to split
        :param X: the predictors (numpy array)
        :param y: the target label (numpy array)
        :return: a dictionary with the best split feature index, threshold, gain, left and right datasets
        """
        # compute number of samples and number of features
        num_features = X.shape[1]

        # initialize the best split
        best_split = {'gain': -1, 'feature': None, 'threshold': None, 'left_dataset': None, 'right_dataset': None}

        # loop over all features (13 in totals)
        for feature_index in range(num_features):
            feature_values = X[:, feature_index]
            thresholds = np.unique(feature_values)

            # loop over all values of the features
            for threshold in thresholds:
                left_mask = feature_values < threshold
                right_mask = feature_values >= threshold
                X_left, y_left = X[left_mask], y[left_mask]
                X_right, y_right = X[right_mask], y[right_mask]

                # check the split result with min_samples constraint
                if len(y_left) >= self.min_samples and len(y_right) >= self.min_samples:
                    if self.criterion == 'entropy':
                        gain = self._information_gain(y, y_left, y_right)
                    elif self.criterion == 'gini':
                        gain = self._gini_gain(y, y_left, y_right)
                    elif self.criterion == 'chi_square':
                        gain = self._chi_square_split(y, y_left, y_right)
                    else:
                        return None

                    # update the best split if the current gain is better
                    if gain > best_split['gain']:
                        best_split['gain'] = gain
                        best_split['feature'] = feature_index
                        best_split['threshold'] = threshold
                        best_split['left_dataset'] = (X_left, y_left)
                        best_split['right_dataset'] = (X_right, y_right)
        return best_split

    def _grow_tree(self, X, y, current_depth=0):
        """
        recursively build the decision tree
        :param X: the predictors (numpy array)
        :param y: the target label (numpy array)
        :param current_depth: the initial depth of the tree
        :return: the decision tree
        """
        total_samples = len(y)
        num_labels = len(np.unique(y))

        # when we reach max_depths and min_samples, we stop splitting
        if num_labels == 1 or current_depth >= self.max_depths or total_samples < self.min_samples:
            leaf_value = np.bincount(y).argmax()  # return the most common labels
            return Node(is_leaf=True, prediction=leaf_value)

        # find best split
        best_split = self._find_best_split(X, y)

        # if it's not a valid split, we stop
        if best_split['gain'] == 0:
            leaf_value = np.bincount(y).argmax()
            return Node(is_leaf=True, prediction=leaf_value)

        # extract left and right datasets
        X_left, y_left = best_split['left_dataset']
        X_right, y_right = best_split['right_dataset']

        # Split the dataset based on the best split
        left_tree = self._grow_tree(X_left, y_left, current_depth + 1)
        right_tree = self._grow_tree(X_right, y_right, current_depth + 1)

        return Node(
            feature=best_split['feature'],
            threshold=best_split['threshold'],
            left=left_tree,
            right=right_tree
        )

    def fit(self, X, y):
        """
        Fit the decision tree model to the training data.
        :param X: the predictors (numpy array)
        :param y: the target label (numpy array)
        """
        self.tree = self._grow_tree(X, y)

    def predict(self, X):
        """
        predict the labels for a set of samples.
        :param X: the predictors (numpy array)
        :return: the predicted labels (numpy array)
        """
        return np.array([self.tree.decide(x) for x in X])

    def zero_one_loss(self, X, y):
        """
        compute the 0-1 loss
        :param X: the predictors (ndarray)
        :param y: the target label (ndarray)
        :return: the error rate (float)
        """
        if len(y) == 0:
            return 0  # No data to evaluate

        predictions = np.array(self.predict(X))
        zero_one_loss = np.mean(predictions != y)  # Compute fraction of incorrect predictions

        return zero_one_loss

    def accuracy(self, y_true, y_pred):
        """
        compute the accuracy of the decision tree classifier
        :param y_true: the true labels (numpy array)
        :param y_pred: the predicted labels (numpy array)
        :return: accuracy value of the model (float)
        """
        y_true, y_pred = np.array(y_true), np.array(y_pred)  # Ensure NumPy arrays
        total_samples = len(y_true)

        if total_samples == 0:
            return 0  # Avoid division by zero

        correct_predictions = np.sum(y_true == y_pred)
        return correct_predictions / total_samples


from joblib import Parallel, delayed
from itertools import product


# function for hyperparameter tuning
def grid_search_cv(classifier, param_grid, X_train, y_train, cv=5, n_jobs=-1):
    best_score = -1
    best_params = None

    # generate all possible parameter combinations
    param_combinations = [dict(zip(param_grid.keys(), values)) for values in product(*param_grid.values())]

    def fold_evaluate(params, i):
        """ run a single fold evaluation for cross-validation"""
        X_train_fold, X_val_fold, y_train_fold, y_val_fold = train_test_split(X_train, y_train, test_size=1 / cv,
                                                                              random_state=i)
        classifier.set_params(**params)
        classifier.fit(X_train_fold, y_train_fold)
        y_pred = classifier.predict(X_val_fold)
        return classifier.accuracy(y_val_fold, y_pred)

    # Iterate through all parameter combinations
    for params in param_combinations:
        scores = Parallel(n_jobs=n_jobs)(delayed(fold_evaluate)(params, i) for i in range(cv))
        mean_score = np.mean(scores)

        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    return best_params, best_score




