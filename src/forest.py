# Random Forest Algorithm on iris data
# referenced https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/

import time
import numpy as np
from src import categutils as cat_util
from src import commons as dut

from src.tree.tree_categorical import DecisionTreeCatgorical

class Forest():
    def __init__(self, tree_num):
        self.tree_num = tree_num
        self.tree_arr = []
        self.res_arr = []
        self.res = []

    def fit(self, X, y):
        """
        For each tree, calls fit
        :param X: we need to work on subsample here
        :param y:
        :return:
        """
        for i in range(self.tree_num):
            tree = DecisionTreeCatgorical().fit(X, y, range(3))
            self.tree_arr.append(tree)

    def predict(self, X):
        """
        Tests the performance of the trees.
        :param X: test set
        :return:
        """
        for tree in self.tree_arr:
            res = tree.predict(X)
            self.res_arr.append(res)

        self.res_arr = np.array(self.res_arr)
        self.res_arr = self.res_arr.astype(int)

        # TODO: potential discuss how to speed this up
        # can try  prof's array indexing tricks??
        for i in range(len(X.index)):
            counts = np.bincount(self.res_arr[:,i])
            self.res.append(np.argmax(counts))

        return self.res





def main():
    forest = Forest(3)

    filename = "../data/iris_data.csv"
    df = cat_util.read_pd(filename)
    # NOTE!!: this must be done, otherwise some strange indexing error in pandas
    train_df, test_df = cat_util.split_train_test(df, train=0.8)
    train_df = train_df.reset_index(drop=True)

    test_df = test_df.reset_index(drop=True)
    X = train_df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    y = train_df["species"]

    start_time = time.time()

    forest.fit(X, y)
    X = test_df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    actual = test_df["species"]
    preds = forest.predict(X)
    print("preds", preds)
    print("--- %s seconds ---" % (time.time() - start_time))

    accuracy = dut.accuracy_metric(actual.values, preds)
    print(accuracy)


if __name__ == '__main__':
    main()
