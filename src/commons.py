# plots, and other functions
# TODO: plot the runtime with different functions, with threading, etc
from csv import reader
from random import randrange
import numpy as np
import pandas as pd

# DATAUTILS
def load_csv(filename):
    dataset = list()
    with open(filename, "r") as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

# Create a random subsample from the dataset with replacement
def subsample(dataset, ratio):
    sample = list()
    n_sample = round(len(dataset) * ratio)
    while len(sample) < n_sample:
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample


def gen_confusion_matrix(true, pred):
    combined = np.append(true, pred)
    combined_codes = pd.Categorical(combined).codes  # map categories to : 0 to n-1
    half = len(combined_codes) // 2  # gives int
    true_i = combined_codes[0:half]
    pred_i = combined_codes[half:len(combined_codes)]

    k = len(np.unique(true_i))  # number of classes
    result = np.zeros((k, k))  # build matrix (not really a matrix, is an array) of K by K
    for i in range(len(true)):  # for every observation
        result[true_i[i]][pred_i[i]] += 1  # add + 1 to the combination
    return result, np.unique(true)
