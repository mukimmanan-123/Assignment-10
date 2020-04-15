import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Logistic Regression Using L1 and L2 Regularization
def Logistic_Regression_Using_Gradient_Descent(lr, weights, epochs, x, y):
    l1 = 0.4
    l2 = 1 - l1
    costs = []
    for _ in range(epochs):
        prediction = np.dot(x, weights)
        sigmoid = 1 / (1 + np.exp(-prediction))
        error = sigmoid - y
        cost = (-y) * np.log(sigmoid) - (1 - y) * np.log(1 - sigmoid)
        cost = sum(cost) / len(x)
        costs.append(cost)
        weights = weights - (lr * (x.T.dot(error) + l1 * np.sign(weights) + l2 * 2 * weights) * 1 / len(x))
    return weights, costs


def accuracy_metric(actual, predicted):
    correct = 0
    for value in range(len(actual)):
        if actual[value] == predicted[value]:
            correct += 1
    return correct / len(actual) * 100.0


np.random.seed(123)
data = pd.read_csv("Logistic_Regression_Dataset.csv")
v = len(data.columns)
X = data.iloc[:, 1:v - 1].values
# thetas = np.random.rand(len(X[0]))
thetas = np.zeros(len(X[0]))
Y = data.iloc[:, v - 1].values

# Splitting Data_Sets
val = len(X) // 3
X_train = X[val:]
X_test = X[: val]
Y_train = Y[val:]
Y_test = Y[: val]

theta_, past_cost = Logistic_Regression_Using_Gradient_Descent(0.025, thetas, 1000, X_train, Y_train)
print(past_cost)
print(theta_)

# Plot the cost function...
plt.title('Cost Function')
plt.xlabel('No. of iterations')
plt.ylabel('Cost')
plt.plot(past_cost)
plt.show()

predictions = []
for i in X_test:
    s = 0
    for j in range(0, len(i)):
        # print(j)
        s = s + (theta_[j] * i[j])
    z = round(1 / (1 + np.exp(-s)))
    predictions.append(z)
print(predictions)
print(Y_test)
print("Accuracy = ", accuracy_metric(Y_test, predictions))
