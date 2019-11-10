# utility functions that will  be phased out
from csv import reader
from random import randrange




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
    def variance_SSR_max_feat(df, feature, target):
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
            if column == 'quality':  # We can't splt on the target variable
                continue
            splits = df[column].unique()  # Possible splits are all the unique values, except the last value (because of <=)
            mean = df[column].mean()
            for split in splits[:-1]:  # We have to exclude the last value as split
                mean_target_group1 = df[df[column] <= split][target].mean()
                mean_target_group2 = df[df[column] > split][target].mean()
                len_group1 = sum(df[column] <= split)
                len_group2 = len(df.index) - len_group1
                variance_SSR = len_group1 * (mean_target_group1 - mean) ** 2 + len_group2 * (
                            mean_target_group2 - mean) ** 2
                # print("variance_SSR", variance_SSR)
                if variance_SSR > max_SSR:
                    best_split = split
                    max_SSR = variance_SSR
                    best_column = column
        return best_split, best_column, max_SSR
