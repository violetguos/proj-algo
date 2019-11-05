# Random Forest Algorithm on Sonar Dataset
# this will be cleaned down to only forest definition
from random import seed
from random import randrange
from math import sqrt
import time
from multiprocessing.pool import Pool
from proj import contiutils as dut


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
            # score the group based on the score for each class
            for class_val in classes:
                p = [row[-1] for row in group].count(class_val) / size
                score += p * p
                # weight the group score by its relative size
            gini += (1.0 - score) * (size / n_instances)
        return gini

    # Select the best split point for a dataset
    def get_split(self, dataset, n_features):
        class_values = list(set(row[-1] for row in dataset))
        b_index, b_value, b_score, b_groups = 999, 999, 999, None
        features = list()
        while len(features) < n_features:
            index = randrange(len(dataset[0]) - 1)
            if index not in features:
                features.append(index)
        for index in features:
            for row in dataset:
                groups = self.test_split(index, row[index], dataset)
                gini = self.gini_index(groups, class_values)
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = (
                        index,
                        row[index],
                        gini,
                        groups,
                    )
        return {"index": b_index, "value": b_value, "groups": b_groups}


    # Make a prediction with a decision tree
    def predict(self, node, row):
        if row[node["index"]] < node["value"]:
            if isinstance(node["left"], dict):
                return self.predict(node["left"], row)
            else:
                return node["left"]
        else:
            if isinstance(node["right"], dict):
                return self.predict(node["right"], row)
            else:
                return node["right"]


    # Create a terminal node value
    def to_terminal(self, group):
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)


    # Create child splits for a node or make terminal
    def split(self, node, max_depth, min_size, n_features, depth):
        left, right = node["groups"]
        del (node["groups"])
        # check for a no split
        if not left or not right:
            node["left"] = node["right"] = self.to_terminal(left + right)
            return
            # check for max depth
        if depth >= max_depth:
            node["left"], node["right"] = self.to_terminal(left), self.to_terminal(right)
            return
            # process left child
        if len(left) <= min_size:
            node["left"] = self.to_terminal(left)
        else:
            node["left"] = self.get_split(left, n_features)
            self.split(node["left"], max_depth, min_size, n_features, depth + 1)
            # process right child
        if len(right) <= min_size:
            node["right"] = self.to_terminal(right)
        else:
            node["right"] = self.get_split(right, n_features)
            self.split(node["right"], max_depth, min_size, n_features, depth + 1)


    # Build a decision tree
    def build_tree(self, train, max_depth, min_size, n_features):
        root = self.get_split(train, n_features)
        self.split(root, max_depth, min_size, n_features, 1)
        return root


class Forest:
    def __init__(self, data):
        self.n_folds = 5
        self.max_depth = 10
        self.min_size = 1
        self.dataset = data

    def predict(self, model):
        pass



    def evaluate_algorithm(self, *args):
        folds = dut.cross_validation_split(self.dataset, self.n_folds)
        scores = list()
        for fold in folds:
            train_set = list(folds)
            train_set.remove(fold)
            train_set = sum(train_set, [])
            test_set = list()
            for row in fold:
                row_copy = list(row)
                test_set.append(row_copy)
                row_copy[-1] = None
            predicted = self.random_forest_seq(train_set, test_set, *args)
            actual = [row[-1] for row in fold]
            accuracy = dut.accuracy_metric(actual, predicted)
            scores.append(accuracy)
        return scores


    # Make a prediction with a list of bagged trees
    def bagging_predict(self, trees, row):
        # TODO: use threads for each tree's result
        t = Tree()
        predictions = [t.predict(tree, row) for tree in trees]
        return max(set(predictions), key=predictions.count)


    # Random Forest Algorithm
    def random_forest(
        self, train, test, max_depth, min_size, sample_size, n_trees, n_features
    ):
        pool = Pool(processes=4)

        trees = list()
        start_time = time.time()
        async_result = list()
        # TODO: replace this loop with threads
        for i in range(n_trees):
            sample = dut.subsample( train, sample_size)
            t = Tree()
            res = pool.apply_async(
                t.build_tree, (sample, max_depth, min_size, n_features)
            )
            async_result.append(res)

        return_val = [res.get(timeout=1) for res in async_result]
        trees = return_val

        # trees = []
        # for i in range(n_trees):

        #   sample = subsample(train, sample_size)
        #   tree = build_tree(sample, max_depth, min_size, n_features)
        #   trees.append(tree)
        # predictions = [bagging_predict(trees, row) for row in test]

        predictions = [self.bagging_predict(trees, row) for row in test]
        print("--- %s seconds ---" % (time.time() - start_time))
        return predictions


    def random_forest_seq(
        self,train, test, max_depth, min_size, sample_size, n_trees, n_features
    ):
        # TODO:add the tress list to self param
        trees = list()
        start_time = time.time()
        # TODO: replace this loop with threads
        for i in range(n_trees):
            sample = dut.subsample(train, sample_size)
            t = Tree()
            tree = t.build_tree(sample, max_depth, min_size, n_features)
            trees.append(tree)
        predictions = [self.bagging_predict(trees, row) for row in test]
        print("--- %s seconds ---" % (time.time() - start_time))

        return predictions


def main():
    # Test the random forest algorithm
    seed(2)
    # load and prepare data
    filename = "sonar.all-data.csv"
    dataset = dut.load_csv(filename)
    # convert string attributes to integers
    for i in range(0, len(dataset[0]) - 1):
        dut.str_column_to_float(dataset, i)
    # convert class column to integers
    dut.str_column_to_int(dataset, len(dataset[0]) - 1)
    # evaluate algorithm
    n_folds = 5
    max_depth = 10
    min_size = 1
    sample_size = 1.0
    n_features = int(sqrt(len(dataset[0]) - 1))
    optim = Forest(dataset)
    for n_trees in [30, 50]:
        scores = optim.evaluate_algorithm(
            max_depth,
            min_size,
            sample_size,
            n_trees,
            n_features,
        )
        print("Trees: %d" % n_trees)
        print("Scores: %s" % scores)
        print("Mean Accuracy: %.3f%%" % (sum(scores) / float(len(scores))))


if __name__ == '__main__':
    main()
