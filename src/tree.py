# a single tree implemented using categorical variables, iris dataset

# reference:  https://levelup.gitconnected.com/building-a-decision-tree-from-scratch-in-python-machine-learning-from-scratch-part-ii-6e2e56265b19
import numpy as np
import pandas as pd
from src import categutils as cat_util
import time
from src import commons as dut


class DecisionTreeCatgorical:

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
        # TODO: adapt this function
        for c in range(self.col_count):
            self.find_better_split(c)
        if self.is_leaf:
            return

        selected_x = self.x.values[self.idxs, self.var_idx]
        lhs = np.nonzero(selected_x <= self.split)[0]
        rhs = np.nonzero(selected_x > self.split)[0]
        self.lhs = Node(self.x, self.y, self.idxs[lhs])
        self.rhs = Node(self.x, self.y, self.idxs[rhs])

    def cat_split(self, var_idx):
        col = self.x.iloc[self.idxs, var_idx]
        unique = col.unique()
        res = list()
        for threshold in unique:
            left, right = list(), list()
            print("type col", type(col))
            for index, col_val in col.items():
                if col_val < threshold:
                    # only append index of that row, not making a copy
                    left.append(index)
                else:
                    right.append(index)
            res.append((left, right, threshold))
        return res

    def find_better_split(self, var_idx):
        x_col = self.x.iloc[self.idxs, var_idx]
        # TODO: don't use flower name as the target var??
        splits = self.cat_split(var_idx)


        test_s_1 = splits[0][0]
        test_s_2 = splits[0][1]
        test_s = test_s_1 + test_s_2

        test_1 = self.idxs.tolist()
        test_s = test_s.sort()
        test_1 = test_1.sort()
        assert test_s == test_1

        best_lhs, best_rhs = None, None
        for split in splits:

            # split is a list of 2 lists, now we assign them to the lhs and rhs

            lhs = pd.Series(split[0])
            rhs = pd.Series(split[1])
            if rhs.sum() < self.min_leaf_count or lhs.sum() < self.min_leaf_count: continue

            curr_score = self.find_score(split)
            if curr_score < self.score:
                self.var_idx = var_idx
                self.score = curr_score
                self.split = split
                best_lhs, best_rhs = lhs, rhs
        # TODO: this only returns the best lhs, rhs of each col feature, not the global one
        return best_lhs, best_rhs




    def find_score(self, split):
        pk = []
        score = 0
        categories = range(2)
        for category in categories:
            sum_cat = 0.0
            subtree = pd.Series(split[0])
            total_rows = subtree.shape[0]
            if len(split[0]) > 0:
                for idx in subtree:
                    print("idx", idx)
                    break
                    if self.y.iloc[idx] == category:
                        sum_cat += 1
                pk.append(sum_cat / len(subtree))  # Find proportion in each class
            score += (1 - sum([p ** 2 for p in pk])) * (len(subtree) / total_rows)
        return score


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

        if self.is_leaf: return self.val
        # else, recurse left and right from current node
        if xi[self.var_idx] <= self.split:
            node = self.lhs
        else:
            node = self.rhs
        return node.predict_col_helper(xi)


filename = "../data/iris_data.csv"
df = cat_util.read_pd(filename)
train_df, test_df = cat_util.split_train_test(df)
print("df shape", df.shape)
# NOTE!!: this must be done, otherwise some strange indexing error in pandas
test_df = df[0:100]
X = train_df[["sepal_length","sepal_width", "petal_length", "petal_width"]]
y = train_df["species"]

start_time = time.time()

regressor = DecisionTreeCatgorical().fit(X, y)
X = test_df[0:50][["sepal_length","sepal_width", "petal_length", "petal_width"]]
actual = test_df[0:50]["species"]
preds = regressor.predict(X)
print("--- %s seconds ---" % (time.time() - start_time))

accuracy = dut.accuracy_metric(actual.values, preds)
print(accuracy)