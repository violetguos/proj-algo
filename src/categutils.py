# utility functions in the oct 13 notebook

import pandas as pd # to read the csv
import random # generate random numbers for train test split
import math # to calculate log2 probabilities
import numpy as np

def read_pd(fp):
    df = pd.read_csv(fp)
    df.head()
    df = df.replace('setosa', 0.0)
    df = df.replace('versicolor', 1.0)
    df = df.replace('virginica', 2.0)
    return df

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


# Function to generate a random sample of n observations from the data set (random sample with replacement)
def bootstrap(df):
    index=[]
    for i in range(0,len(df)):# for number of rows in df:
        index.append(my_randrange(len(df))) # generate boostrap indexes
    return df.iloc[index].reset_index(drop=True) # return df rows that correspond to the random sample + reset index


# Alternate function that allows to change the sampling percentage
#(so not all observations are chosen: implies speedup possibilities, but performance can suffer)
def bootstrap_sample(df, sample=1.0):
    index=[]
    for i in range(0,round(len(df) * sample)): # for number of rows in df x sample %
        index.append(my_randrange(len(df))) # generate boostrap indexes
    return df.iloc[index].reset_index(drop=True) # return df rows that correspond to the random sample + reset index


# Select a random sample of the columns (columns cannot repeat, so sampling without replacement)
def column_sample(df, sample=1.0):
    index=[]
    i = 1
    while i<=round(df.shape[1] * sample): # for number of columns in df x sample %
        index_to_add = my_randrange(df.shape[1]) # random number (number of columns)
        if index_to_add not in index: # don't add a column if it's already in the sample
            index.append(index_to_add)
            i = i+1
    return df.iloc[:,index] # return df rows that correspond to the random sample + reset index


class CategoricalUtil:

    # First step is to find the pk
    @staticmethod
    def pk(df, target):
        categories = df[target].unique()
        pk = []
        i = 0
        for category in categories:
            pk.append(sum(df[target] == category) / len(df))
            i = i + 1
        return categories, pk

    # Gini function
    @staticmethod
    def gini(df, categories):
        # most up to date gini function used in training
        # categories = df[target].unique()
        total_rows = float(sum([len(subtree) for subtree in df]))

        pk = []
        score = 0
        for subtree in df:
            # only 2. either left or right
            for category in categories:
                sum_cat = 0.0
                if len(subtree) > 0:
                    for row in subtree:
                        if row[-1] == category:
                            sum_cat += 1
                    pk.append(sum_cat/len(subtree))  # Find proportion in each class
            score += (1-sum([p ** 2 for p in pk])) * (len(subtree)/ total_rows)
        return score

    # Entropy function
    @staticmethod
    def entropy(df, target):
        categories = df[target].unique()
        pk = []
        i = 0
        for category in categories:
            pk.append(sum(df[target] == category) / len(df))  # Find proportion in each class
            i = i + 1
        return -1 * sum([p * math.log2(p) for p in pk])  # Return entropy

    @staticmethod
    def cat_split(df, target):
        unique = df[target].unique()
        for val in unique:
            left, right = list(), list()
            for row in df:
                if row[index] < value:
                    left.append(row)
                else:
                    right.append(row)
            return left, right


    @staticmethod
    def cat_split_old(df, target):
        # TODO: use the updated split method instead of a plain recursion

        unique = df[target].unique()
        # print(unique.size)
        if unique.size == 1:
            return  # This is not really what we want, need to change this
            # We will have to change it to, if unique.size=1, then don<t split on this variable.
        else:  # if size >=2:
            l = [[unique[0]], [unique[1]]]
            if unique.size == 2:
                return l  # only one possible split
            else:
                # n=3:
                left = unique[0]
                right = unique[1]
                l = [left, right]
                result = []
                result.append([[unique[2]]] + [l])  # 3, [1,2] - so putting 3 rd value alone

                result.append([[unique[2]] + [l[0]]] + [[l[1]]])  # putting 3rd value on left side

                result.append([[l[0]]] + [[unique[2]] + [l[1]]])  # putting 3rd value on right side
                if unique.size == 3:
                    return result

                for i in range(3, unique.size):  # for n = 4 to n max, equivalent to for i=3 to n-1
                    l = result
                    result = []  # to only return the splits with all the categories

                    # adding new number alone to the left and putting the rest to the right
                    result.append([[unique[i]]] + [l[0][0] + l[0][1]])

                    for j in range(0, 2 ** (i - 1) - 1):
                        result.append([[unique[i]] + l[j][0]] + [l[j][1]])  # adding number to left
                        result.append([l[j][0]] + [[unique[i]] + l[j][1]])  # adding number to the right

                return result