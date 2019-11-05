# utility functions in the oct 13 notebook

import pandas as pd # to read the csv
import random # generate random numbers for train test split
import math # to calculate log2 probabilities

def read_pd(fp="winequality-red.csv"):
    df = pd.read_csv(fp, sep=';')
    return df

def split_train_test(df, train=0.60):
    train_size = round(len(df) * train)
    train_indices = random.sample(population=df.index.tolist(), k=train_size)
    train_df = df.loc[train_indices]
    test_df = df.loc[set(df.index) - set(train_df.index)] #get rest of index
    return train_df, test_df

######## For continuous ##########
class ContinousUtil:
    @staticmethod
    def variance_SSR(df, feature, split, target):
        """
        for continuous target
        Split criteria based on reduction of variance (so maximising SSR)
        attention group 1 is <= split !!!
        :param df:
        :param feature:
        :param split:
        :param target:
        :return:
        """
        mean_target_group1 = df[df[feature]<=split][target].mean()
        len_group1 = sum(df[feature]<=split)
        mean_target_group2 = df[df[feature]>split][target].mean()
        len_group2 = len(df.index) - len_group1
        mean = df[target].mean()
        variance_SSR = len_group1 * (mean_target_group1 - mean)**2 + len_group2 * (mean_target_group2 - mean)**2
        return variance_SSR

    @staticmethod
    def variance_SSR_max(df, feature, target):
        """
        Alternate function, find max of SSR for all possible splits for a specific variable
        :param df:
        :param feature:
        :param target:
        :return:
        """
        splits = df[feature].unique() # all possible splits are all unique values, except the first value
        max_SSR = 0
        mean = df[target].mean()
        for split in splits[1:]: # We have to exclude the first value as split is <=
            mean_target_group1 = df[df[feature]<=split][target].mean()
            mean_target_group2 = df[df[feature]>split][target].mean()
            len_group1 = sum(df[feature]<=split)
            len_group2 = len(df.index) - len_group1
            variance_SSR = len_group1 * (mean_target_group1 - mean)**2 + len_group2 * (mean_target_group2 - mean)**2
            if variance_SSR > max_SSR:
                best_split = split
                max_SSR=variance_SSR
        return best_split

    @staticmethod
    def variance_SSR_max(df, target):
        """
        Alternate function to find the best split out of all possible splits (so for all variables):

        :param df:
        :param target:
        :return:
        """
        max_SSR = 0
        for column in df:
            if column == 'quality': # We can't splt on the target variable
                continue
            splits = df[column].unique() # Possible splits are all the unique values, except the last value (because of <=)
            mean = df[column].mean()
            for split in splits[:-1]: # We have to exclude the last value as split
                mean_target_group1 = df[df[column]<=split][target].mean()
                mean_target_group2 = df[df[column]>split][target].mean()
                len_group1 = sum(df[column]<=split)
                len_group2 = len(df.index) - len_group1
                variance_SSR = len_group1 * (mean_target_group1 - mean)**2 + len_group2 * (mean_target_group2 - mean)**2
                if variance_SSR > max_SSR:
                    best_split = split
                    max_SSR = variance_SSR
                    best_column = column
        return best_split, best_column

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
    def gini(df, target):
        categories = df[target].unique()
        pk = []
        i = 0
        for category in categories:
            pk.append(sum(df[target] == category) / len(df))  # Find proportion in each class
            i = i + 1
        return 1 - sum([p ** 2 for p in pk])  # Return gini

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
            for next_one in cat_split(categories[1:]):  # need to exclude first category, as stored in 'first'
                for i, subset in enumerate(next_one):
                    yield next_one[:i] + [[first] + subset] + next_one[i + 1:]
                yield [[first]] + next_one