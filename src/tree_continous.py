# tree def on the wine data
# reference:  https://levelup.gitconnected.com/building-a-decision-tree-from-scratch-in-python-machine-learning-from-scratch-part-ii-6e2e56265b19
import numpy as np
import pandas as pd
from src import categutils as cat_util
import time


class DecisionTreeRegressor:
    # TODO: borrowed
    def fit(self, X, y, min_leaf=5):
        self.dtree = Node(X, y, np.array(np.arange(len(y))), min_leaf)
        return self

    def predict(self, X):
        return self.dtree.predict(X.values)


class Node:

    def __init__(self, x, y, idxs, min_leaf_count=5):
        self.x = x
        self.y = y
        self.idxs = idxs
        self.min_leaf_count = min_leaf_count
        self.row_count = len(idxs)
        self.col_count = x.shape[1]
        self.val = np.mean(y[idxs])
        self.score = float('inf')
        self.find_varsplit()

    def find_varsplit(self):
        # TODO: borrowed
        # TODO: consider the missing values, if any
        for c in range(self.col_count): self.find_better_split(c)
        if self.is_leaf: return

        selected_x = self.x.values[self.idxs, self.var_idx]
        lhs = np.nonzero(selected_x <= self.split)[0]
        rhs = np.nonzero(selected_x > self.split)[0]
        self.lhs = Node(self.x, self.y, self.idxs[lhs])
        self.rhs = Node(self.x, self.y, self.idxs[rhs])

    def find_better_split(self, var_idx, all=False):
        # TODO: borrowed
        if all is True:
            x = self.x.values[self.idxs, var_idx]

            for r in range(self.row_count):
                # NOTE: minor improvement by not comparing all the rows
                lhs = x <= x[r]
                rhs = x > x[r]
                if rhs.sum() < self.min_leaf_count or lhs.sum() < self.min_leaf_count: continue

                curr_score = self.find_score(lhs, rhs)
                if curr_score < self.score:
                    self.var_idx = var_idx
                    self.score = curr_score
                    self.split = x[r]
        else:
            x = self.x.iloc[self.idxs, var_idx]
            splits = x.unique()
            print(splits)

            for split in splits:
                lhs = x <= split
                rhs = x > split
                if rhs.sum() < self.min_leaf_count or lhs.sum() < self.min_leaf_count: continue

                curr_score = self.variance_ssr(lhs, rhs)
                if curr_score < self.score:
                    self.var_idx = var_idx
                    self.score = curr_score
                    self.split = split


    def find_score(self, lhs, rhs):
        # TODO: get rid this, already have var_ssr
        y_select = self.y[self.idxs]
        # print("y sel", y_select)
        # print("lhs", lhs)
        lhs_std = y_select[lhs].std()
        rhs_std = y_select[rhs].std()
        return lhs_std * lhs.sum() + rhs_std * rhs.sum()

    def variance_ssr(self, lhs, rhs):
        y_select = self.y[self.idxs]
        mean_target_group1 = y_select[lhs].mean()
        len_group1 = len(lhs)
        mean_target_group2 = y_select[rhs].mean()
        len_group2 = len(rhs)
        mean_y = y_select.mean()
        variance_SSR = len_group1 * (mean_target_group1 - mean_y)**2 + len_group2 * (mean_target_group2 - mean_y)**2
        return variance_SSR

    @property
    def is_leaf(self):
        return self.score == float('inf')

    def predict(self, x):
        return np.array([self.predict_col_helper(xi) for xi in x])

    def predict_col_helper(self, xi):
        """
        Recurse helper
        :param xi: each column in the df
        :return:
        """
        if self.is_leaf:
            return self.val

        # else, recurse left and right from current node
        if xi[self.var_idx] <= self.split:
            node = self.lhs
        else:
            node = self.rhs
        return node.predict_col_helper(xi)


df = pd.read_csv("../data/winequality-red.csv", sep=';')
train_df, test_df = cat_util.split_train_test(df)
# NOTE!!: this must be done, otherwise some strange indexing error in pandas
test_df = df[0:50]
X = test_df[["fixed acidity", "density"]]
print("type", type(X))
y = test_df["quality"]

start_time = time.time()

regressor = DecisionTreeRegressor().fit(X, y)
X = train_df[0:10]
preds = regressor.predict(X)
print(preds)
print("--- %s seconds ---" % (time.time() - start_time))
