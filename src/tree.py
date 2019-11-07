# Random Forest Algorithm on Sonar Dataset
# this will be cleaned down to only forest definition
from random import seed
from random import randrange
from math import sqrt
import time
from multiprocessing.pool import Pool
from src import categutils as cat_util
from src import commons as dut
import pandas as pd

class Node:
    def __init__(self, val):
        self.val = val

class Tree:
    def __init__(self):
        self.feature = 1
    
    # Split a dataset based on an attribute and an attribute value
    def test_split(self, index, value, dataset):
        left, right = list(), list()
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right

    # Calculate the Gini index for a split dataset
    def gini_index(self, groups, classes):
        # NOTE: this is before cleaning
        # new updated function is in categutils.py

        # count all samples at split point
        n_instances = float(sum([len(group) for group in groups]))
        # sum weighted Gini index for each group
        gini = 0.0
        for group in groups:
            size = float(len(group))
            # avoid divide by zero
            if size == 0:
                continue
            score = 0.0
            # TODO add the gini index helper from C here
            # score the group based on the score for each class
            for class_val in classes:
                p = [row[-1] for row in group].count(class_val) / size
                score += p * p
                # weight the group score by its relative size
            gini += (1.0 - score) * (size / n_instances)
        return gini

    # Select the best split point for a dataset
    def get_split(self, dataset, n_features):
        #TODO: use C's imp
        class_values = list(set(row[-1] for row in dataset))
        # depends whether subset data, this could be subset of 0,1,2
        b_index, b_value, b_score, b_groups = 999, 999, 999, None
        features = list()
        while len(features) < n_features:
            index = randrange(len(dataset[0]) - 1)
            if index not in features:
                features.append(index)
        for index in features:
            for row in dataset:
                # row[index] is the threshold value?
                groups = self.test_split(index, row[index], dataset)
                # print(groups)
                # self.gini_index(groups, class_values)
                gini = cat_util.CategoricalUtil.gini(groups, class_values)
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = (
                        index,
                        row[index],
                        gini,
                        groups,
                    )
        return Node({"index": b_index, "value": b_value, "groups": b_groups})


    # Make a prediction with a decision tree
    def predict(self, node, row):
        if row[node.val["index"]] < node.val["value"]:
            if isinstance(node.val["left"], Node):
                return self.predict(node.val["left"], row)
            else:
                return node.val["left"]
        else:
            if isinstance(node.val["right"], Node):
                return self.predict(node.val["right"], row)
            else:
                return node.val["right"]


    # Create a terminal node value
    def to_terminal(self, group):
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)


    # Create child splits for a node or make terminal
    def split(self, node, max_depth, min_size, n_features, depth):
        left, right = node.val["groups"]
        del (node.val["groups"])
        # check for a no split
        if not left or not right:
            node.val["left"] = node.val["right"] = self.to_terminal(left + right)
            return
            # check for max depth
        if depth >= max_depth:
            node.val["left"], node.val["right"] = self.to_terminal(left), self.to_terminal(right)
            return
            # process left child
        if len(left) <= min_size:
            node.val["left"] = self.to_terminal(left)
        else:
            node.val["left"] = self.get_split(left, n_features)
            self.split(node.val["left"], max_depth, min_size, n_features, depth + 1)
            # process right child
        if len(right) <= min_size:
            node.val["right"] = self.to_terminal(right)
        else:
            node.val["right"] = self.get_split(right, n_features)
            self.split(node.val["right"], max_depth, min_size, n_features, depth + 1)


    # Build a decision tree
    def build_tree(self, train, max_depth, min_size, n_features):
        root = self.get_split(train, n_features)
        self.split(root, max_depth, min_size, n_features, 1)
        return root


class Forest:
    def __init__(self, train_data, test_data):
        self.n_folds = 5
        self.max_depth = 10
        self.min_size = 1
        # self.dataset = data
        self.train_data = train_data
        self.test_data = test_data

    def predict(self, model):
        pass

    def evaluate_algorithm(self, parallel, *args):
        scores = []
        if parallel is False:
            print("Test sequential")
            predicted = self.random_forest_seq(*args)
        else:
            print("Test parallel")
            predicted = self.random_forest(*args)
        actual = [row[-1] for row in self.test_data]
        accuracy = dut.accuracy_metric(actual, predicted)
        scores.append(accuracy)
        return scores

    # Make a prediction with a list of bagged trees
    def bagging_predict(self, trees, row):
        t = Tree()
        predictions = [t.predict(tree, row) for tree in trees]
        return max(set(predictions), key=predictions.count)

    # Random Forest Algorithm
    def random_forest(
        self, max_depth, min_size, sample_size, n_trees, n_features
    ):
        pool = Pool(processes=4)

        trees = list()
        start_time = time.time()
        async_result = list()
        for i in range(n_trees):
            sample = dut.subsample(self.train_data, sample_size)
            t = Tree()
            res = pool.apply_async(
                t.build_tree, (sample, max_depth, min_size, n_features)
            )
            async_result.append(res)

        return_val = [res.get(timeout=1) for res in async_result]
        trees = return_val


        predictions = [self.bagging_predict(trees, row) for row in self.test_data]
        print("--- %s seconds ---" % (time.time() - start_time))
        return predictions

    def random_forest_seq(
        self, max_depth, min_size, sample_size, n_trees, n_features
    ):
        # TODO:add the tress list to self param
        trees = list()
        start_time = time.time()
        for i in range(n_trees):
            sample = dut.subsample(self.train_data, sample_size)
            t = Tree()
            tree = t.build_tree(sample, max_depth, min_size, n_features)
            print("*"*20)
            print(tree.val)
            print("*"*20)
            trees.append(tree)
        predictions = [self.bagging_predict(trees, row) for row in self.test_data]
        print("--- %s seconds ---" % (time.time() - start_time))
        return predictions


def main():
    # Test the random forest algorithm
    seed(2)
    # load and prepare data
    filename = "../data/iris_data.csv"
    dataset = cat_util.read_pd(filename)

    t1, t2 = cat_util.split_train_test(dataset)

    train_data = t1.values
    test_data = t2.values
    #
    # evaluate algorithm
    n_folds = 5
    max_depth = 10
    min_size = 1
    sample_size = 1.0
    n_features = 4 # 4 for iris
    x = dataset.values
    print(type(x))
    optim = Forest(train_data, test_data)
    for n_trees in [3, 5]:
        scores = optim.evaluate_algorithm(
            False, # seq
            max_depth,
            min_size,
            sample_size,
            n_trees,
            n_features,
        )
        print("Trees: %d" % n_trees)
        print("Scores: %s" % scores)
        print("Mean Accuracy: %.3f%%" % (sum(scores) / float(len(scores))))

    # for n_trees in [30, 50]:
    #     scores = optim.evaluate_algorithm(
    #         True, # parallel
    #         max_depth,
    #         min_size,
    #         sample_size,
    #         n_trees,
    #         n_features,
    #     )
    #     print("Trees: %d" % n_trees)
    #     print("Scores: %s" % scores)
    #     print("Mean Accuracy: %.3f%%" % (sum(scores) / float(len(scores))))
    #

if __name__ == '__main__':
    main()
