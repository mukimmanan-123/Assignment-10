from sklearn import datasets
import pandas as pd


def Data_Generation_For_K_Means(no_of_values, features):
    x, y = datasets.make_blobs(no_of_values, features, random_state=0, centers=3)
    cols = []
    for i in range(features):
        cols.append(i)
    data_set = pd.DataFrame(x, columns=cols)
    data_set.to_csv("Data_Set_K_Means.csv")


Data_Generation_For_K_Means(1000, 2)
