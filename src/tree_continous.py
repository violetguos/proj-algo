# definition of all the forest functions

# the continous case
from random import seed
from random import randrange
from math import sqrt
import time
from multiprocessing.pool import Pool
from src import categutils as cat_util
from src import contiutils as cont_util
from src import commons as dut
import pandas as pd


def build_tree(df):
    left = 0
    right = 0
    left_leaves_count = 0
    right_leaves_count = 0
    split = cont_util.ContinousUtil.variance_SSR_max(df, 'quality')
    for index, row in df.iterrows():
        if row[split[1]] < split[0]:
            # the target val
            left += row[-1]
            left_leaves_count +=1
        else:
            right += row[-1]
            right_leaves_count +=1
    return left/left_leaves_count, right/right_leaves_count

def main():
    df = pd.read_csv("../data/winequality-red.csv", sep=';')
    train_df, test_df = cat_util.split_train_test(df)
    test_df = df[1:10]
    x = build_tree(test_df)
    print(x)



if __name__ == '__main__':
    main()
