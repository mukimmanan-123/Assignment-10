import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Linear Regression Using L1 and L2 Regularization
def L1_L2_Regularization_Linear_Regression(x, y, epochs, lr, weight):
    l1 = 0.4
    l2 = 1 - l1
    costs = []
    for _ in range(epochs):
        prediction = np.dot(x, weight)
        error = prediction - y
        cost = 1 / (len(x)) * np.dot(error.T, error)
        costs.append(cost)
        weight = weight - (lr * (x.T.dot(error) + l1 * np.sign(weight) + l2 * 2 * weight) * 1 / len(x))
        print(weight)
    return costs, weight


def MSE(y, predicted):
    error = y - predicted
    return 1 / (len(y)) * np.dot(error.T, error)


np.random.seed(123)
data = pd.read_csv("Linear_Regression_Dataset.csv")
v = len(data.columns)
X = data.iloc[:, 1:v - 1].values
thetas = np.random.randn(len(X[0]) + 1)
print(thetas)
# Appending 1 for intercept purpose
X = np.c_[np.ones(X.shape[0]), X]
Y = data.iloc[:, v - 1].values

# Splitting Data_Sets
val = len(X) // 3
X_train = X[val:]
X_test = X[: val]
Y_train = Y[val:]
Y_test = Y[: val]
cost_val, weights = L1_L2_Regularization_Linear_Regression(X_train, Y_train, 1000, 0.025, thetas)
plt.title('Cost Function')
plt.xlabel('No. of iterations')
plt.ylabel('Cost')
plt.plot(cost_val)
plt.show()

predictions = []
c = weights[0]
for i in X_test:
    s = 0
    for j in range(0, len(i) - 1):
        # print(j)
        s = s + (weights[j + 1] * i[j + 1])
    s += c
    predictions.append(s)

print(predictions)
print(Y_test)
print("MSE : ", MSE(Y_test, predictions))
