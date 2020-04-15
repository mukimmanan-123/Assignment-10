import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt


# K - Means (Using Random Centroids) From Scratch
def K_Means(x, no_of_clusters):
    centroids = np.array([]).reshape(len(x[0]), 0)
    # print(centroids)
    for i in range(no_of_clusters):
        random_val = random.randint(0, len(x) - 1)
        centroids = np.c_[centroids, x[random_val]]
    print(centroids)
    distance = euclidean_distance(centroids, no_of_clusters, len(x), x)
    plt.scatter(x[:, 0], x[:, 1], c='maroon')
    plt.show()
    color = ['red', 'blue', 'green', 'black', 'purple', 'orange', 'brown', 'pink', 'neon']
    if len(x[0]) == 2:
        for k in range(no_of_clusters):
            plt.scatter(x=distance[k][:, 0], y=distance[k][:, 1], c=color[k])
        plt.scatter(centroids[0, :], centroids[1, :], color='yellow')
        plt.show()
    for va in distance.keys():
        print("cluster {} -----> {}".format(va, distance[va]))
    return centroids, distance


def euclidean_distance(centroids, no_of_clusters, length, x):
    euclidean = np.array([]).reshape(length, 0)
    for i in range(no_of_clusters):
        distance = np.sum((x - centroids[:, i]) ** 2, axis=1)
        euclidean = np.c_[euclidean, distance]

    # print(euclidean)
    index = np.argmin(euclidean, axis=1)
    # print(index)
    distances = {}
    for k in range(no_of_clusters):
        distances[k] = np.array([]).reshape(len(x[0]), 0)
    for i in range(length):
        distances[index[i]] = np.c_[distances[index[i]], x[i]]
    for k in range(no_of_clusters):
        distances[k] = distances[k].T
    return distances


def weighted_sum_of_squares(x):
    weighted_sum_squares = []
    for va in range(1, 8):
        centroids, distance = K_Means(x, va)
        weighted_sum_square = 0
        for k in range(va):
            weighted_sum_square += np.sum((distance[k] - centroids[:, k]) ** 2)
        weighted_sum_squares.append(weighted_sum_square)

    plt.plot(weighted_sum_squares)
    plt.show()


data = pd.read_csv("Data_Set_K_Means.csv")
val = len(data.columns)
print(val)
X = data.iloc[:, 1: val].values
Y = data.iloc[:, val - 1].values
print(len(X[0]))
K_Means(X, 3)
# weighted_sum_of_squares(X)
