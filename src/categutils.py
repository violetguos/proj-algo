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

