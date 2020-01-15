#  Random forest project for HEC's algorithm class

TLDR: we built a random forest regressor and classifier from scratch, without any sklearn.

Functionalities include:

* Load predefined datasets (found on the popular UCI repo)
* Modular decision tree, can be used as a predictor/regressor by itself
* A `forest` class that aggregates result by averaging or voting

Result:
* Comparable benchmark against the standard `RandomForest` in scikit learn, at least on the few datasets we tried
* Note that we didn't run a statistical test. However, you may judge for yourself whether that's significant (Referece: Figueiredo Filho, Dalson Britto et al. (2013). “When is statistical significance not significant?” In:Brazilian Political Science Review7.1, pp. 31–55)
