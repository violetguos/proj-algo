# Random Forest Algorithm on iris data
# referenced https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/

import time
import numpy as np
from src import commons as dut

from src.tree.tree_categorical import DecisionTree, check_col_type
import pandas as pd
from src.const import STR_CATEGORICAL, STR_CONTINUOUS

class Forest():
    def __init__(self, tree_num, sampler, column_names, target_name, x_type, y_type, func='entropy'):
        """
        :param tree_num: this defined how many trees we train in a RF algo
        :param tree_arr: this stores a list root nodes to trees in the forsest
        :param res_arr: an array of `tree_num` arrays, where each array stores the prediction of
        one tree. Then we apply majority function to make a final decisoion

        Illustation:
        This is res_arr, it's dim n by m, where n = tree_num, m depends on the test data dimension
        test data
           x1, .... xm
        [ [ res of tree 1]
          [ y1 .... ym   ]
          [ res of tree 2]
          ....
          [ res of tree n]
        ]
          At this vertical axis, we make a decision for each X.

        self.res is a list of dimension 1 by m.

        """
        self.tree_num = tree_num
        self.sampler = sampler
        self.tree_arr = []
        self.res_arr = []
        self.res = []
        self.column_names = column_names
        self.target_name = target_name
        self.x_type = x_type
        self.y_type = y_type
        self.func = func

    def fit(self, df, labels=range(3)):
        """
        For each tree, calls Node.fit
        :param X: we need to work on subsample here
        :param y: label
        :return:
        """

        for i in range(self.tree_num):
            df_subset = self.sampler(df, sample=0.9)
            col_indices = dut.column_sample(df_subset[self.column_names], sample=0.6)
            x_type = check_col_type(df_subset)
            x_type = x_type[0:-1]
            print(col_indices)
            X = df_subset
            y = df_subset[self.target_name]
            tree = DecisionTree().fit(X, x_type, col_indices, y, self.y_type, labels,  min_leaf=5, func=self.func)
            self.tree_arr.append(tree)

    def predict(self, X):
        """
        Tests the performance of the trees.
        :param X: test set
        :return: the result array
        """
        for tree in self.tree_arr:
            res = tree.predict(X)
            self.res_arr.append(res)

        if self.y_type == STR_CATEGORICAL:
            self.res_arr = np.array(self.res_arr)
            # self.res_arr = self.res_arr.astype(int)

            for i in range(len(X.index)):
                # this `[:,i]` array indexing enables us to look at each column
                # this is the majority count of a 2D matrix in np array format
                items, counts = np.unique(self.res_arr[:,i], return_counts=True)
                # counts = np.bincount(self.res_arr[:,i])
                self.res.append(items[np.argmax(counts)])
        else:
            self.res_arr = np.array(self.res_arr)
            # print("res arr shape", self.res_arr[:,0].shape)
            for i in range(len(X.index)):
                self.res.append(np.mean(self.res_arr[:,i]))


        return self.res

def main():

    # 3 is very arbitary here. Only for testing.
    column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    target_name = "species"
    print("Experiment using row sampling")


    filename = "../data/iris_data.csv"
    df = pd.read_csv(filename)
    train_df, test_df = dut.split_train_test(df, train=0.8)
    train_df = train_df.reset_index(drop=True)
    x_col_types = check_col_type(train_df)
    print(x_col_types)
    test_df = test_df.reset_index(drop=True)

    start_time = time.time()
    forest = Forest(50, dut.bootstrap_sample, column_names, target_name, x_col_types, STR_CATEGORICAL, func='entropy')

    forest.fit(train_df)

    X = test_df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    actual = test_df["species"]
    preds = forest.predict(X)
    print("preds", preds)
    print("--- %s seconds ---" % (time.time() - start_time))
    accuracy = dut.accuracy_metric(actual.values, preds)
    print(accuracy)




def main_conti():
    print("Experiment using row sampling")
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
    target_name = "MEDV"

    filename = "../data/housing.csv"
    df = pd.read_csv(filename, sep=r"\s+")
    # NOTE!!: this must be done, otherwise some strange indexing error in pandas
    train_df, test_df = dut.split_train_test(df, train=0.8)
    train_df = train_df.reset_index(drop=True)

    test_df = test_df.reset_index(drop=True)
    x_col_types = check_col_type(train_df)
    forest = Forest(10, dut.bootstrap_sample, column_names, target_name, x_col_types, STR_CONTINUOUS)

    start_time = time.time()

    forest.fit(train_df)

    X = test_df[column_names]
    actual = test_df["MEDV"]
    preds = forest.predict(X)
    print("--- %s seconds ---" % (time.time() - start_time))

    accuracy = dut.mse_metric(actual.values, preds)

    print("mse", accuracy)


if __name__ == '__main__':
    # main()
    main_conti()
