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
        print("np.array(np.arange(len(y)))", np.array(np.arange(len(y))))
        self.dtree = Node(X, y, np.array(np.arange(len(y))), min_leaf)
        return self

    def predict(self, X):
        return self.dtree.predict(X.values)


class Node:

    def __init__(self, x, y, idxs, min_leaf_count=5):
        assert(x.shape[1]==4)
        print("x.shape[1]", x.shape[1])
        self.x = x
        self.y = y
        self.idxs = idxs
        self.min_leaf_count = min_leaf_count
        self.row_count = len(idxs)
        self.col_count = x.shape[1] - 1
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
        # lhs = np.nonzero(selected_x <= self.split)[0]
        # rhs = np.nonzero(selected_x > self.split)[0]
        self.lhs = Node(self.x, self.y, self.best_lhs_indices)
        self.rhs = Node(self.x, self.y, self.best_rhs_indices)

    def cat_split(self, var_idx):
        assert(var_idx < 4)

        try:
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
                res.append((left, right, threshold))
                return res

        except Exception as e:


            print(e)
            print("self.idxs", self.idxs)
            print("var_idx", var_idx)

    def find_better_split(self, var_idx):

        # x_col = self.x.iloc[self.idxs, var_idx]
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
                # NOTE: this is the line that makes it the global
                self.score = curr_score
                self.split = split
                self.best_lhs_indices, self.best_rhs_indices = lhs, rhs




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
                    # print("idx", idx)
                    # print("sleg y", self.y)
                    if self.y[idx] == category:
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
        :param xi: each row in the df
        :return:
        """

        if self.is_leaf:
            return self.val
        # else, recurse left and right from current node
        if xi[self.var_idx] <= self.split[2]:
            node = self.lhs
        else:
            node = self.rhs
        return node.predict_col_helper(xi)


filename = "../data/iris_data.csv"
df = cat_util.read_pd(filename)
# train_df, test_df = cat_util.split_train_test(df)
print("df shape", df.shape)
# NOTE!!: this must be done, otherwise some strange indexing error in pandas
X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y = df["species"]

print(X)
print(X.shape)
print(X.iloc[5])
start_time = time.time()

regressor = DecisionTreeCatgorical().fit(X, y)
X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
actual = df["species"]
preds = regressor.predict(X)
print("preds", preds)
print("--- %s seconds ---" % (time.time() - start_time))

accuracy = dut.accuracy_metric(actual.values, preds)
print(accuracy)
