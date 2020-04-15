import pandas as pd
from scipy.stats import norm
import numpy as np


def Data_Generation_For_Linear_Regression(no_of_rows, no_of_attributes):
    error = norm.rvs(0, 0.25, no_of_rows)
    # print(error)
    features = []
    for v in range(no_of_attributes):
        x = norm.rvs(0, 1, no_of_rows)
        x = x[:, np.newaxis]
        if v == 0:
            features = x
        else:
            features = np.c_[x, features]
    print(features)

    y = []
    for value in features:
        summation = 0
        for w in range(0, len(value)):
            print(value[w])
            summation = summation + (1 / (w + 2)) * value[w]
            print(summation)
        y.append(summation)
    y = y + error - 1
    cols = []
    for i in range(no_of_attributes):
        cols.append(i)
    data_set = pd.DataFrame(features, columns=cols)
    data_set["y"] = y
    print(data_set.head())
    data_set.to_csv("Linear_Regression_Dataset.csv")


Data_Generation_For_Linear_Regression(1000, 2)
