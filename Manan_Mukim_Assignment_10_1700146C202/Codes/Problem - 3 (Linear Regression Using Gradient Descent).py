import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Linear Regression Using Gradient Descent
def Linear_Regression_Using_Batch_Gradient_Descent(lr, theta, epochs, x, y):
    costs = []
    for _ in range(epochs):
        prediction = np.dot(x, theta)
        error = prediction - y
        cost = 1 / (len(x)) * np.dot(error.T, error)
        costs.append(cost)
        theta = theta - (lr * (1 / len(x)) * np.dot(x.T, error))
        print(theta)
    return theta, costs


def MSE(y, predicted):
    error = y - predicted
    return 1 / (len(y)) * np.dot(error.T, error)


np.random.seed(123)
data = pd.read_csv("Linear_Regression_Dataset.csv")
v = len(data.columns)
X = data.iloc[:, 1:v - 1].values
thetas = np.random.rand(len(X[0]) + 1)

# Appending 1 for intercept purpose
X = np.c_[np.ones(X.shape[0]), X]
Y = data.iloc[:, v - 1].values

# Splitting Data_Sets
val = len(X) // 3
X_train = X[val:]
X_test = X[: val]
Y_train = Y[val:]
Y_test = Y[: val]

theta_, past_cost = Linear_Regression_Using_Batch_Gradient_Descent(0.021, thetas, 1000, X_train, Y_train)
print(past_cost)
print(theta_)

# Plot the cost function...
plt.title('Cost Function')
plt.xlabel('No. of iterations')
plt.ylabel('Cost')
plt.plot(past_cost)
plt.show()

predictions = []
c = theta_[0]
for i in X_test:
    s = 0
    for j in range(1, len(i)):
        # print(j)
        s = s + (theta_[j] * i[j])
    s += c
    predictions.append(s)

print(predictions)
print(Y_test)
print("MSE : ", MSE(Y_test, predictions))
