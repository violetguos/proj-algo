# a single tree implemented using categorical variables, iris dataset

# reference:  https://levelup.gitconnected.com/building-a-decision-tree-from-scratch-in-python-machine-learning-from-scratch-part-ii-6e2e56265b19
import numpy as np
import pandas as pd
from src import categutils as cat_util
import time
from src import commons as dut


# TODO: self.split has a tuple of 3 variables, (left, right, threshold)
# Should make it into another class?? or not jsut to make things sper clear
# instead of indexing [0][1[2]

class DecisionTreeCatgorical:

    def fit(self, X, y, min_leaf=5):
        self.dtree = Node(X, y, np.array(np.arange(len(y))), min_leaf)
        return self

    def predict(self, X):
        return self.dtree.predict(X.values)


class Split:
    def __init__(self, lhs, rhs, threshold):
        self.lhs = lhs
        self.rhs = rhs
        self.threshold = threshold

    def print(self):
        print("lhs {} rhs {} threshold".format(self.lhs, self.rhs, self.threshold))


class Node:

    def __init__(self, x, y, idxs, min_leaf_count=5):
        self.x = x
        self.y = y
        self.idxs = idxs
        self.min_leaf_count = min_leaf_count
        self.row_count = len(idxs)
        self.col_count = x.shape[1]
        # the self.val is a majority vote
        self.val = y[idxs].mode()[0]
        self.score = float('inf')
        self.find_varsplit()

    def find_varsplit(self):
        for c in range(self.col_count):
            self.find_better_split(c)
        if self.is_leaf:
            return

        self.lhs = Node(self.x, self.y, self.best_lhs_indices)
        self.rhs = Node(self.x, self.y, self.best_rhs_indices)

    def cat_split(self, var_idx):
        # this handles some indexing exceptions
        # no longer happens as of commit PR #2

        col = self.x.iloc[self.idxs, var_idx]

        unique = col.unique()
        res = list()
        for threshold in unique:
            left, right = list(), list()
            # print("type col", type(col))
            for index, col_val in col.items():
                if col_val < threshold:
                    # only append index of that row, not making a copy
                    left.append(index)
                else:
                    right.append(index)
        res.append(Split(left, right, threshold))
        return res



    def find_better_split(self, var_idx):

        splits = self.cat_split(var_idx)

        for split in splits:
            # split is a list of 2 lists, now we assign them to the lhs and rhs

            lhs = pd.Series(split.lhs)
            rhs = pd.Series(split.rhs)
            if rhs.sum() < self.min_leaf_count or lhs.sum() < self.min_leaf_count:
                continue

            curr_score = self.find_score(split)
            if curr_score < self.score:
                self.var_idx = var_idx
                # NOTE: this is the line that makes it the global score
                self.score = curr_score
                self.split = split
                self.best_lhs_indices, self.best_rhs_indices = lhs, rhs

    def find_score(self, split):
        # this is the old gini function
        pk = []
        score = 0
        # only 0, 1, 2 in iris dataset
        categories = range(3)
        subtrees = split.lhs, split.rhs
        for category in categories:
            sum_cat = 0.0
            for s in subtrees: # left or right, 0 for left, 1 for right
                subtree = pd.Series(s)
                total_rows = subtree.shape[0]
                if total_rows > 0:
                    for idx in subtree:
                        if self.y[idx] == category:
                            sum_cat += 1
                    pk.append(sum_cat / len(subtree))  # Find proportion in each class
                score += (1 - sum([p ** 2 for p in pk])) * (len(subtree) / total_rows)
        return score


    @property
    def is_leaf(self):
        return self.score == float('inf')

    def predict(self, x):
        # predict row by row
        res = []
        for row in x:
            res.append(self.predict_row_helper(row))
        return np.array(res)

    def predict_row_helper(self, row):
        """
        Recurse helper
        :param row: each row in the df
        :return:
        """

        if self.is_leaf:
            return self.val
        # else, recurse left and right from current node
        if row[self.var_idx] <= self.split.threshold:
            node = self.lhs
        else:
            node = self.rhs
        return node.predict_col_helper(row)


def main():
    filename = "../data/iris_data.csv"
    df = cat_util.read_pd(filename)
    # NOTE!!: this must be done, otherwise some strange indexing error in pandas
    X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]][0:140]
    y = df["species"][0:140]

    start_time = time.time()

    regressor = DecisionTreeCatgorical().fit(X, y)
    X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]][140:150]
    actual = df["species"][140:150]
    preds = regressor.predict(X)
    print("preds", preds)
    print("--- %s seconds ---" % (time.time() - start_time))

    accuracy = dut.accuracy_metric(actual.values, preds)
    print(accuracy)


if __name__ == '__main__':
    main()

