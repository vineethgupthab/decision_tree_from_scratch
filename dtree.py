import numpy as np
from scipy import stats
from sklearn.metrics import r2_score, accuracy_score

class DecisionNode:
    def __init__(self, col, split, lchild, rchild):
        self.col = col
        self.split = split
        self.lchild = lchild
        self.rchild = rchild

    def predict(self, x_test):
        # Make decision based upon x_test[col] and split
        if x_test[self.col] > self.split:
            return self.rchild.predict(x_test)
        else:
            return self.lchild.predict(x_test)


class LeafNode:
    def __init__(self, y, prediction):
        "Create leaf node from y values and prediction; prediction is mean(y) or mode(y)"
        self.n, self.prediction = y.shape[0], prediction

    def predict(self, x_test):
        # return prediction
        return self.prediction


def gini(x):
    """
    Return the gini impurity score for values in y
    Assume y = {0,1}
    Gini = 1 - sum_i p_i^2 where p_i is the proportion of class i in y
    """
    if x.shape[0]<1:
        return 0
    class_freq, class_proportions = dict(), dict()
    for cls in np.unique(x):
        class_freq[cls] = x[x==cls].shape[0]
        class_proportions[cls] = class_freq[cls]/x.shape[0]
    return 1- np.sum(np.power(np.array(list(class_proportions.values())),2))


def find_best_split(X, y, loss, min_samples_leaf):
    best_gini_score, best_split, w_gini = float('inf'), None, float('inf')

    n_features, col = X.shape[1], 0

    while col < n_features:
        vals = np.unique(X[:, col])
        i = 0

        while i < len(vals) - 1:
            split_value = (vals[i] + vals[i + 1]) / 2

            indices_left = X[:, col] <= split_value
            X_left, y_left = X[indices_left], y[indices_left]
            X_right, y_right = X[~indices_left], y[~indices_left]

            if y_left.shape[0] >= min_samples_leaf or y_right.shape[0] >= min_samples_leaf:
                left_side_gini, right_side_gini = loss(y_left), loss(y_right)
                samples = y_left.shape[0] + y_right.shape[0]
                w_gini = (y_left.shape[0] / samples) * left_side_gini + (y_right.shape[0] / samples) * right_side_gini

            if w_gini >= best_gini_score:
                pass
            else:
                best_gini_score = w_gini
                best_split = (col, split_value)
            i += 1
        
        col += 1

    return best_split
    
    
class DecisionTree621:
    def __init__(self, min_samples_leaf=1, loss=None):
        self.min_samples_leaf = min_samples_leaf
        self.loss = loss # loss function; either np.var for regression or gini for classification
        
    def fit(self, X, y):
        """
        Create a decision tree fit to (X,y) and save as self.root, the root of
        our decision tree, for  either a classifier or regression.  Leaf nodes for classifiers
        predict the most common class (the mode) and regressions predict the average y
        for observations in that leaf.

        This function is a wrapper around fit_() that just stores the tree in self.root.
        """
        self.root = self.fit_(X, y)


    def fit_(self, X, y):
        """
        Recursively create and return a decision tree fit to (X,y) for
        either a classification or regression.  This function should call self.create_leaf(X,y)
        to create the appropriate leaf node, which will invoke either
        RegressionTree621.create_leaf() or ClassifierTree621.create_leaf() depending
        on the type of self.

        This function is not part of the class "interface" and is for internal use, but it
        embodies the decision tree fitting algorithm."""
        
        if y.shape[0] <= self.min_samples_leaf or np.unique(y).shape[0] == 1:
            return self.create_leaf(y)
        else:
            best_split = find_best_split(X, y, self.loss, self.min_samples_leaf)
            col, split = best_split[0], best_split[1]

            if col == -1:
                return self.create_leaf(y)

            X_left, y_left = X[X[:, col] <= split], y[X[:, col] <= split]
            X_right, y_right = X[X[:, col] > split], y[X[:, col] > split]


            if y_left.shape[0] == 0 or y_right.shape[0] == 0:
                return self.create_leaf(y)

            lchild = self.fit_(X_left, y_left)
            rchild = self.fit_(X_right, y_right)
            return DecisionNode(col, split, lchild, rchild)

    
    def predict(self, X_test):
        """
        Make a prediction for each record in X_test and return as an array.
        This method is inherited by RegressionTree621 and ClassifierTree621 and
        works for both without modification!
        """
        def traverse_tree(x, node):
            if hasattr(node, 'prediction'):
                return node.prediction
            return traverse_tree(x, node.rchild) if x[node.col] > node.split else traverse_tree(x, node.lchild)

        predictions = np.array(list(map(lambda x: traverse_tree(x, self.root), X_test)))
        return predictions


class RegressionTree621(DecisionTree621):
    def __init__(self, min_samples_leaf=1):
        super().__init__(min_samples_leaf, loss=np.var)

    def score(self, X_test, y_test):
        "Return the R^2 of y_test vs predictions for each record in X_test"
        return r2_score(y_test, self.predict(X_test))

    def create_leaf(self, y):
        """
        Return a new LeafNode for regression, passing y and mean(y) to
        the LeafNode constructor.
        """
        return LeafNode(y, np.mean(y))


class ClassifierTree621(DecisionTree621):
    def __init__(self, min_samples_leaf=1):
        super().__init__(min_samples_leaf, loss=gini)

    def score(self, X_test, y_test):
        "Return the accuracy_score() of y_test vs predictions for each record in X_test"
        return accuracy_score(y_test, self.predict(X_test))

    def create_leaf(self, y):
        """
        Return a new LeafNode for classification, passing y and mode(y) to
        the LeafNode constructor. Feel free to use scipy.stats to use the mode function.
        """
        return LeafNode(y, stats.mode(y)[0])
