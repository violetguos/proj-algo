from csv import reader
from random import randrange
import numpy as np
import pandas as pd
import random # generate random numbers for train test split

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

# Calculate accuracy percentage
def mse_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        correct += (actual[i] - predicted[i]) ** 2
    return correct / len(actual)


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

def split_train_test(df, train=0.60):
    train_size = round(len(df) * train)
    train_indices = random.sample(population=df.index.tolist(), k=train_size)
    train_df = df.loc[train_indices]
    test_df = df.loc[set(df.index) - set(train_df.index)] #get rest of index
    return train_df, test_df

def remove_missing_target(df, target):
    """
    Function to remove all rows having a missing target (index is reset)
    :param df: the original dataframe
    :param target: key of the target column
    :return:
    """
    return df.dropna(subset=[target]).reset_index(drop=True)

def my_randrange(n): # starts at 0 included ends at n-1 included
    return int(random.random()* n // 1) # returns floor of random(0,1) * n


# Alternate function that allows to change the sampling percentage
#(so not all observations are chosen: implies speedup possibilities, but performance can suffer)
def bootstrap_sample(df, sample=1.0):
    index=[]
    for i in range(0,round(len(df) * sample)): # for number of rows in df x sample %
        index.append(my_randrange(len(df))) # generate boostrap indexes
    return df.iloc[index].reset_index(drop=True) # return df rows that correspond to the random sample + reset index


# Select a random sample of the columns (columns cannot repeat, so sampling without replacement)
def column_sample(df, sample=0.6):
    index=[]
    i = 0
    print("df.shape[1]", df.shape[1])
    while i < round(df.shape[1] * sample): # for number of columns in df x sample %
        index_to_add = my_randrange(df.shape[1]) # random number (number of columns)
        if index_to_add not in index: # don't add a column if it's already in the sample
            index.append(index_to_add)
            i = i + 1
    index.sort()
    return index #, np.array(x_types)[index] # return df rows that correspond to the random sample + reset index

