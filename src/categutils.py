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
            return unique  # This is not really what we want, need to change this
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