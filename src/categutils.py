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
    def cat_split(categories):
        # Next we need to find all possible binary splits
        #### does not work with return, only works with yield, why?
        # Bacically the idea here is to generate the subsets (possible splits) in a smart way.
        # This is a recursive algo. Basically, you start with the first category. Call this left.
        # Next you take the second category and set it aside. Call this right.
        # The way to create a subset of size n when you have all the subsets of size (n-1) is to add the
        # new element to the left, then to the right, and then create a new subset with it on the left.
        # ex for n=3:
        # 1
        # 1 2
        # 13  2
        # 1   23
        # 3   12
        # for n=4:
        # same as n=3 plus:
        # 134  2
        # 13   24
        # 14   23
        # 1    234
        # 34   12
        # 3    124
        # 4    123
        if len(categories) == 1:
            yield [categories]
        else:
            first = categories[0]
            for next_one in CategoricalUtil.cat_split(categories[1:]):  # need to exclude first category, as stored in 'first'
                for i, subset in enumerate(next_one):
                    yield next_one[:i] + [[first] + subset] + next_one[i + 1:]
                yield [[first]] + next_one
