# a single tree implemented using categorical variables, iris dataset

# reference:  https://levelup.gitconnected.com/building-a-decision-tree-from-scratch-in-python-machine-learning-from-scratch-part-ii-6e2e56265b19
import numpy as np
import pandas as pd
from src import categutils as cat_util
import time
from src import commons as dut

import math


class DecisionTreeCatgorical:

    def fit(self, X, y, label_set, min_leaf=5):
        self.dtree = Node(X, y, np.array(np.arange(len(y))),label_set, min_leaf)
        return self

    def predict(self, X):
        return self.dtree.predict(X.values)


class Split:
    def __init__(self, lhs, rhs, threshold):
        """
        # self.split has a tuple of 3 variables, (left, right, threshold)
        Should make it into another class?, jsut to make things SUPER clear
        instead of indexing [0][1[2]
        :param lhs:
        :param rhs:
        :param threshold:
        """
        self.lhs = lhs
        self.rhs = rhs
        self.threshold = threshold

    def print(self):
        """
        This prints helpful information in intermediate steps
        :return:
        """
        print("lhs {} rhs {} threshold".format(self.lhs, self.rhs, self.threshold))


class Node:

    def __init__(self, x, y, idxs, label_set, min_leaf_count=5):

        """
        :param x: data column
        :param y: the target variable
        :param idxs: the subset at this node's split
        :param min_leaf_count: prevents overfitting
        :param val: a majority vote based on all the Ys in this node
        :param label_set: the set of all labels in the dataset, deisgned to adapt to different datasets
        """
        self.x = x
        self.y = y
        self.idxs = idxs

        self.min_leaf_count = min_leaf_count
        self.row_count = len(idxs)
        self.col_count = x.shape[1]
        self.val = y[idxs].mode()[0]
        self.score = float('inf')
        self.label_set = label_set


        # this method is automatically called when we declare a node.
        # only a short hand notation
        self.find_varsplit()

    def find_varsplit(self):
        """
        Iterate all the columns
        Find the best one to split one at a time
        :return:
        """

        # before trying to split, we see if this node is already pure
        curr_unique = self.y.unique()
        if len(curr_unique) == 1:
            self.score = float('inf')

        for c in range(self.col_count):
            self.find_better_split(c)
        if self.is_leaf:
            return

        # after we found the newest LHS, and RHS, declare the new nodes
        # and store them inside self parameters
        # so we can run a prediction outside the scope by following these pointers
        self.lhs = Node(self.x, self.y, self.best_lhs_indices, self.label_set)
        self.rhs = Node(self.x, self.y, self.best_rhs_indices, self.label_set)


    def find_pure_in_splits(self, split):
        """
        A helper function for the find_better_split()
        If a subtree contains pure y labels,  we don't check the rest
        :return:
        """
        lhs_unique = self.y[split.lhs].unique()

        #print(lhs_unique)
        # print(len(lhs_unique))
        rhs_unique = self.y[split.rhs].unique()
        if len(lhs_unique) == 1 or len(rhs_unique) == 1:
            return True
        else:
            return False



    def find_better_split(self, var_idx):
        # this generates all the splits
        # TODO: optimize this funciton
        splits = self.cat_split(var_idx)


        # while not res
        # iter in list of splits
        res = False # inital val
        i = 0

        while not res and i < len(splits):
            res = self.find_pure_in_splits(splits[i])
            i += 1

        if res == True and i != len(splits)-1 and len(splits[i-1].lhs) >= self.min_leaf_count and len(splits[i-1].rhs) >= self.min_leaf_count:
            i -= 1
            self.var_idx = var_idx
            curr_score = self.find_score(splits[i])
            # NOTE: this is the line that makes it the global score
            self.score = curr_score
            self.split = splits[i]
            self.best_lhs_indices, self.best_rhs_indices = splits[i].lhs, splits[i].rhs

        else:
            for split in splits:
                # split has 2 lists, now we assign them to the lhs and rhs
                # we store the indices of rows in a given `column[var_idx]`
                lhs = pd.Series(split.lhs)
                rhs = pd.Series(split.rhs)

                if rhs.size < self.min_leaf_count or lhs.size < self.min_leaf_count:
                    continue

                curr_score = self.find_score(split)
                if curr_score < self.score:
                    self.var_idx = var_idx
                    # NOTE: this is the line that makes it the global score
                    self.score = curr_score
                    self.split = split
                    self.best_lhs_indices, self.best_rhs_indices = lhs, rhs

    def cat_split(self, var_idx):
        """
        :param var_idx: the current column from def find_better_split(self)
        :return: all possible splits at this
        """
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

    def find_score(self, split, func='entropy'):
        if func == 'gini':
            return self.gini(split)
        elif func == 'entropy':
            return self.entropy(split)

    def entropy(self, split):

        score = 0
        # only 0, 1, 2 in iris dataset
        categories = self.label_set
        subtrees = split.lhs, split.rhs

        i = 0
        two_subtrees_entropy = 0
        for category in categories:
            pk = []

            for s in subtrees:
                # left or right
                subtree = pd.Series(s)
                # print("self.y[subtree]", self.y[subtree])
                pk.append(sum(self.y[subtree] == category) / len(subtree))  # Find proportion in each class
                # calculate a single entropy, handle the 0 case
                entropy = 0
                for p in pk:
                    if p != 0:
                        entropy += p * math.log2(p)
                entropy = -1 * entropy
            two_subtrees_entropy += entropy

        return  two_subtrees_entropy # Return entropy


    def gini(self, split):
        # this is the old gini function
        pk = []
        score = 0
        # only 0, 1, 2 in iris dataset
        categories = self.label_set
        subtrees = split.lhs, split.rhs
        for category in categories:
            sum_cat = 0.0
            for s in subtrees:  # left or right, 0 for left, 1 for right
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
        return node.predict_row_helper(row)


def main():
    filename = "../../data/iris_data.csv"
    df = cat_util.read_pd(filename)

    # data cleaning
    df = cat_util.remove_missing_target(df, 'species')

    train_df, test_df = cat_util.split_train_test(df, train=0.8)
    train_df = train_df.reset_index(drop=True)

    test_df = test_df.reset_index(drop=True)
    X = train_df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    y = train_df["species"]

    start_time = time.time()

    regressor = DecisionTreeCatgorical().fit(X, y, range(3))
    X = test_df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    actual = test_df["species"]
    preds = regressor.predict(X)
    print("preds", preds)
    print("--- %s seconds ---" % (time.time() - start_time))

    accuracy = dut.accuracy_metric(actual.values, preds)
    print(accuracy)


if __name__ == '__main__':
    main()

