# a single tree implemented using categorical variables, iris dataset

# reference:  https://levelup.gitconnected.com/building-a-decision-tree-from-scratch-in-python-machine-learning-from-scratch-part-ii-6e2e56265b19
import numpy as np
import pandas as pd
from src import categutils as cat_util
import time
from src import commons as dut
import math


def fast_log(x):
    """
    Tayloer series
    𝑓(𝑥)=ln(𝑥)=(𝑥−1)−12(𝑥−1)2+13(𝑥−1)3−14(𝑥−1)4+⋯
    :return: log_2(x)
    """
    ln_2 = 0.693147
    ln_x = (x-1) - 0.5 * (x-1)*(x-1) + (x-1)* (x-1)*(x-1)/3 - (x-1)*(x-1)*(x-1)*(x-1) * 0.25

    log_2_x = ln_x / ln_2

    return log_2_x


def merge_list(l1, l2):
    """
    Merge two lists or two items into a list, or an item and a list into a list, not a list of lists
    :param l1:
    :param l2:
    :return:
    """
    if isinstance(l1, list) and isinstance(l2, list):
        merged_list = l1 + l2
    elif not isinstance(l1, list)  and isinstance(l2, list):
        merged_list = [l1] + l2
    elif isinstance(l1, list) and not isinstance(l2, list):
        merged_list = l1 + [l2]
    elif not isinstance(l1, list) and not isinstance(l2, list):
        merged_list = [l1] + [l2]
    return merged_list


STR_CONTINUOUS = 'continuous'
STR_CATEGOTICAL = 'categorical'

class DataType:
    def __init__(self, data_type):
        self.data_type = data_type

    @property
    def continous(self):
        return self.data_type == STR_CONTINUOUS

    @property
    def categorical(self):
        return self.data_type == STR_CATEGOTICAL


class DecisionTreeCatgorical:

    def fit(self, X, x_type, y, y_type, label_set, min_leaf=5):
        self.dtree = Node(X, x_type, y, y_type, np.array(np.arange(len(y))),label_set, min_leaf)
        return self

    def predict(self, X):
        return self.dtree.predict(X.values)


class SplitContinuous:
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
        Overloading the print function
        This prints helpful information in intermediate steps
        :return:
        """
        print("lhs {} rhs {} threshold".format(self.lhs, self.rhs, self.threshold))


class SplitCategorical:
    def __init__(self, lhs, rhs):

        self.lhs = lhs
        self.rhs = rhs
        self.print()

    def print(self):
        print(self.lhs)
        print(self.rhs)


class Node:

    def __init__(self, x, x_data_type, y, y_data_type, idxs, label_set, min_leaf_count=5):

        """
        :param x: data column
        :param y: the target variable
        :param idxs: the subset at this node's split
        :param min_leaf_count: prevents overfitting
        :param val: a majority vote based on all the Ys in this node
        :param label_set: the set of all labels in the dataset, deisgned to adapt to different datasets
        """
        self.x = x
        self.x_data_type = x_data_type
        self.y = y
        self.y_data_type = y_data_type
        self.idxs = idxs
        self.split_method = self.cat_split if x_data_type == STR_CATEGOTICAL else self.continuous_split


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
        self.lhs = Node(self.x, self.x_data_type, self.y, self.y_data_type, self.best_lhs_indices, self.label_set)
        self.rhs = Node(self.x, self.x_data_type, self.y, self.y_data_type, self.best_rhs_indices, self.label_set)

    def find_pure_in_splits(self, split):
        """
        A helper function for the find_better_split()
        If a subtree contains pure y labels,  we don't check the rest
        :return:
        """
        lhs_unique = self.y[split.lhs].unique()

        # print(lhs_unique)
        # print(len(lhs_unique))
        rhs_unique = self.y[split.rhs].unique()
        if len(lhs_unique) == 1 or len(rhs_unique) == 1:
            return True
        else:
            return False

    def cat_split(self, var_idx):
        # TODO: use the updated split method instead of a plain recursion
        # print(var_idx)
        unique = self.x.iloc[self.idxs, var_idx].unique()
        # print(unique.size)
        if unique.size == 1:
            pass  # This is not really what we want, need to change this
            # We will have to change it to, if unique.size=1, then don<t split on this variable.
        else:  # if size >=2:
            l = [[unique[0]], [unique[1]]]
            if unique.size == 2:

                return [SplitCategorical(l[0], l[1])] # only one possible split
            else:
                # n=3:
                left = unique[0]
                right = unique[1]
                l = [left, right]
                result = []
                result.append([[unique[2]], l])  # 3, [1,2] - so putting 3 rd value alone
                result.append([merge_list(unique[2], l[0]), [l[1]]])  # putting 3rd value on left side
                result.append([[l[0]], merge_list(unique[2], l[1])]) # putting 3rd value on right side


                final_result = []
                if unique.size == 3:
                    # convert to the TYPE split_Categorical
                    for r in result:
                        # i will iterate the left and the right
                        lhs, rhs = [], []
                        res1, res2 = r
                        # for i in r:
                        # this will iterate the whole table
                        for j, val in self.x.iloc[self.idxs, var_idx].items():
                            if val in res1:
                                lhs.append(j)
                            elif val in res2:
                                rhs.append(j)

                        # now convert the lhs and rhs into pd series
                        # lhs = pd.Series(lhs)
                        # rhs = pd.Series(rhs)
                        split = SplitCategorical(lhs, rhs)
                        final_result.append(split)

                    return final_result
                print("unique.size\n", unique.size)
                for i in range(3, unique.size):  # for n = 4 to n max, equivalent to for i=3 to n-1
                    l = result
                    result = []  # to only return the splits with all the categories

                    # adding new number alone to the left and putting the rest to the right
                    result.append([[unique[i]], merge_list(l[0][0], l[0][1])])

                    for j in range(0, 2 ** (i - 1) - 1):
                        result.append([merge_list(unique[i], l[j][0]), [l[j][1]]]) # adding number to left
                        result.append([[l[j][0]], merge_list(unique[i], l[j][1])])  # adding number to the right


                final_result = []

                for r in result:
                    # i will iterate the left and the right
                    lhs, rhs = [], []
                    res1, res2 = r
                    # for i in r:
                    # this will iterate the whole table
                    for j, val in self.x.iloc[self.idxs, var_idx].items():
                        if val in res1:
                            lhs.append(j)
                        elif val in res2:
                            rhs.append(j)

                    # now convert the lhs and rhs into pd series
                    # lhs = pd.Series(lhs)
                    # rhs = pd.Series(rhs)
                    split = SplitCategorical(lhs, rhs)
                    final_result.append(split)
                return final_result

    def find_better_split(self, var_idx):
        # this generates all the splits
        # splits = self.continuous_split(var_idx)

        #  TODO: still testing cat split
        splits = self.split_method(var_idx)
        #splits = self.cat_split(var_idx)

        if splits is None:
            # no more split in categorical X variable
            self.score = float('inf')
            return


        # while not res
        # iter in list of splits
        res = False # inital val
        i = 0

        while not res and i < len(splits):
            res = self.find_pure_in_splits(splits[i])
            i += 1

        # # TODO: do we really need i != len(splits)-1
        # if res is True and i != len(splits)-1 and len(splits[i-1].lhs) >= self.min_leaf_count and len(splits[i-1].rhs) >= self.min_leaf_count:
        #     i -= 1
        #     self.var_idx = var_idx
        #     curr_score = self.find_score(splits[i])
        #     # NOTE: this is the line that makes it the global score
        #     self.score = curr_score
        #     self.split = splits[i]
        #     self.best_lhs_indices, self.best_rhs_indices = splits[i].lhs, splits[i].rhs


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

    def continuous_split(self, var_idx):
        # TODO: move this under class conti split
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
                # take care of the nan values
                if col_val == np.nan:
                    rand = np.random.uniform()
                    if rand < 0.5:
                        left.append(index)
                    else:
                        right.append(index)
                elif col_val < threshold:
                    # only append index of that row, not making a copy
                    left.append(index)
                else:
                    right.append(index)
        res.append(SplitContinuous(left, right, threshold))
        return res

    def variance_ssr(self, lhs, rhs):
        y_select = self.y[self.idxs]
        mean_target_group1 = y_select[lhs].mean()
        len_group1 = len(lhs)
        mean_target_group2 = y_select[rhs].mean()
        len_group2 = len(rhs)
        mean_y = y_select.mean()
        variance_SSR = len_group1 * (mean_target_group1 - mean_y)**2 + len_group2 * (mean_target_group2 - mean_y)**2
        return variance_SSR

    def find_score(self, split, func='entropy'):
        if self.y_data_type == STR_CONTINUOUS:
            return self.variance_ssr(split.lhs, split.rhs)

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
                        entropy += p * fast_log(p)  #math.log2(p)
                entropy = -1 * entropy
            two_subtrees_entropy += entropy

        return two_subtrees_entropy # Return entropy


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
    filename = "../../data/test_cart.csv"
    df = cat_util.read_pd(filename)

    # data cleaning
    df = cat_util.remove_missing_target(df, 'species')

    train_df, test_df = cat_util.split_train_test(df, train=0.5)
    train_df = train_df.reset_index(drop=True)

    test_df = test_df.reset_index(drop=True)
    X = train_df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    y = train_df["species"]
    print(X)

    start_time = time.time()

    regressor = DecisionTreeCatgorical().fit(X, STR_CONTINUOUS, y, STR_CONTINUOUS, range(3))
    X = test_df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    actual = test_df["species"]
    preds = regressor.predict(X)
    print("preds", preds)
    print("--- %s seconds ---" % (time.time() - start_time))

    accuracy = dut.accuracy_metric(actual.values, preds)
    print(accuracy)


if __name__ == '__main__':
    main()

