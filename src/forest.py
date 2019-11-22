# Random Forest Algorithm on iris data
# referenced https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/

import time
import numpy as np
from src import categutils as cat_util
from src import commons as dut

from src.tree.tree_categorical import DecisionTreeCatgorical


class Forest():
    def __init__(self, tree_num, sampler):
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

    def fit(self, df):
        """
        For each tree, calls Node.fit
        :param X: we need to work on subsample here
        :param y: label
        :return:
        """

        for i in range(self.tree_num):
            df_subset = self.sampler(df, sample=0.7)
            X = df_subset[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
            y = df_subset["species"]
            tree = DecisionTreeCatgorical().fit(X, y, range(3))
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

        self.res_arr = np.array(self.res_arr)
        self.res_arr = self.res_arr.astype(int)

        # TODO: potential discuss how to speed this up
        # can try  prof's array indexing tricks??
        for i in range(len(X.index)):
            # this `[:,i]` array indexing enables us to look at each column
            # this is the majority count of a 2D matrix in np array format
            counts = np.bincount(self.res_arr[:,i])
            self.res.append(np.argmax(counts))

        return self.res

def main():

    # 3 is very arbitary here. Only for testing.

    print("Experiment using row sampling")
    forest = Forest(5, cat_util.bootstrap_sample)

    filename = "../data/iris_data.csv"
    df = cat_util.read_pd(filename)
    # NOTE!!: this must be done, otherwise some strange indexing error in pandas
    train_df, test_df = cat_util.split_train_test(df, train=0.8)
    train_df = train_df.reset_index(drop=True)

    test_df = test_df.reset_index(drop=True)
    # X = train_df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    # y = train_df["species"]

    start_time = time.time()

    forest.fit(train_df)

    X = test_df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    actual = test_df["species"]
    preds = forest.predict(X)
    print("preds", preds)
    print("--- %s seconds ---" % (time.time() - start_time))

    accuracy = dut.accuracy_metric(actual.values, preds)
    print(accuracy)


if __name__ == '__main__':
    main()
