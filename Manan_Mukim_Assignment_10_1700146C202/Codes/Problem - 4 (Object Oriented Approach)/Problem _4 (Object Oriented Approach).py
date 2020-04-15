import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Regression:
    def __init__(self, epochs, lr, l1_ratio=None):
        self.epochs = epochs
        self.learning_rate = lr
        self.costings = []
        self.predictions = []
        self.l1_ratio = l1_ratio

    @staticmethod
    def Mean_Squared_Error(actual, predicted):
        error = actual - predicted
        return 1 / (len(actual)) * np.dot(error.T, error)

    @staticmethod
    def accuracy_metric(actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / len(actual) * 100.0

    def gradient_descent(self, x, y, weights, switch=0):
        for _ in range(self.epochs):
            prediction = np.dot(x, weights)
            if switch == 0:
                pass
            else:
                # print("Log")
                prediction = Regression.sigmoid_activation(prediction)
            error = prediction - y
            if switch == 0:
                # print("Linear")
                cost = 1 / (len(x)) * np.dot(error.T, error)
            else:
                # print("Log-1")
                cost = Regression.log_loss(y, prediction)
            self.costings.append(cost)
            if self.l1_ratio is None:
                weights = weights - (self.learning_rate * (1 / len(x)) * np.dot(x.T, error))
            else:
                weights = weights - (self.learning_rate * (x.T.dot(error) + self.l1_ratio * np.sign(weights) + (1 - self.l1_ratio) * 2 * weights) * 1 / len(x))
            # print(weights)
        return weights

    @staticmethod
    def log_loss(y, prediction):
        cost = (-y) * np.log(prediction) - \
               (1 - y) * np.log(1 - prediction)
        cost = sum(cost) / len(y)
        return cost

    @staticmethod
    def sigmoid_activation(prediction):
        return 1 / (1 + np.exp(-prediction))

    def cost_plot(self):
        plt.title('Cost Function')
        plt.xlabel('No. of iterations')
        plt.ylabel('Cost')
        plt.plot(self.costings)
        plt.show()

    def prediction(self, weights, x):
        print(len(x))
        for i in x:
            s = 0
            for j in range(0, len(i)):
                s = s + (weights[j] * i[j])
            self.predictions.append(s)
        return self.predictions


class Linear_Regression(Regression):
    def __init__(self, epochs, lr, x, y, l1_ratio=None):
        super().__init__(epochs=epochs, lr=lr, l1_ratio=l1_ratio)
        self.x = x
        self.y = y
        np.random.seed(1)
        self.weights = np.random.randn(len(x[0] + 1))

    def fit(self):
        self.weights = super().gradient_descent(x=self.x, y=self.y, weights=self.weights)

    def predict(self, x_test):
        self.weights = list(self.weights)
        c = self.weights.pop(0)
        self.predictions = super().prediction(weights=self.weights, x=x_test)
        self.predictions += c
        return self.predictions


class Logistic_Regression(Regression):
    def __init__(self, epochs, lr, x, y, l1_ratio=None):
        super().__init__(epochs=epochs, lr=lr, l1_ratio=l1_ratio)
        self.x = x
        self.y = y
        np.random.seed(1)
        self.weights = np.zeros(len(x[0]))

    def fit(self):
        self.weights = super().gradient_descent(x=self.x, y=self.y, weights=self.weights, switch=1)

    def predict(self, x_test):
        self.predictions = super().prediction(weights=self.weights, x=x_test)
        self.predictions = [round(Regression.sigmoid_activation(value)) for value in self.predictions]
        return self.predictions


# Linear Regression Using Gradient Descent
print("=" * 40)
print("Linear Regression Using Gradient Descent")
data = pd.read_csv("E:\\Machine Learning  Project\\College\\Assignment - 10\\Codes\\Linear_Regression_Dataset.csv")
v = len(data.columns)
X = data.iloc[:, 1:v - 1].values
Y = data.iloc[:, v - 1].values

# Splitting Data_Sets
val = len(X) // 3
X_train = X[val:]
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = X[: val]
Y_train = Y[val:]
Y_test = Y[: val]

model = Linear_Regression(1000, 0.021, X_train, Y_train)
model.fit()
model.cost_plot()
predictions = model.predict(X_test)
print("Predictions")
print(predictions)
print("Original Values")
print(Y_test)
print("Means Squared Error : ", Regression.Mean_Squared_Error(Y_test, predictions))

# Logistic Regression Using Gradient Descent
print("=" * 40)
print("Linear Regression Using Gradient Descent")
data = pd.read_csv("E:\\Machine Learning  Project\\College\\Assignment - 10\\Codes\\Logistic_Regression_Dataset.csv")
v = len(data.columns)
X = data.iloc[:, 1:v - 1].values
Y = data.iloc[:, v - 1].values

# Splitting Data_Sets
val = len(X) // 3
X_train = X[val:]
X_test = X[: val]
Y_train = Y[val:]
Y_test = Y[: val]

model = Logistic_Regression(1000, 0.021, X_train, Y_train)
model.fit()
model.cost_plot()
predictions = model.predict(X_test)
print("Predictions")
print(predictions)
print("Original Values")
print(Y_test)
print("Accuracy Score : ", Regression.accuracy_metric(Y_test, predictions))


# Linear Regression With L1_L2_Regularization
print("=" * 40)
print("Linear Regression With L1 And L2 Regularization")
data = pd.read_csv("E:\\Machine Learning  Project\\College\\Assignment - 10\\Codes\\Linear_Regression_Dataset.csv")
v = len(data.columns)
X = data.iloc[:, 1:v - 1].values
Y = data.iloc[:, v - 1].values

# Splitting Data_Sets
val = len(X) // 3
X_train = X[val:]
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = X[: val]
Y_train = Y[val:]
Y_test = Y[: val]

model = Linear_Regression(1000, 0.021, X_train, Y_train, l1_ratio=0.5)
model.fit()
model.cost_plot()
predictions = model.predict(X_test)
print("Predictions")
print(predictions)
print("Original Values")
print(Y_test)
print("Means Squared Error : ", Regression.Mean_Squared_Error(Y_test, predictions))

# # Logistic Regression With L1_L2_Regularization
data = pd.read_csv("E:\\Machine Learning  Project\\College\\Assignment - 10\\Codes\\Logistic_Regression_Dataset.csv")
print("=" * 40)
print("Logistic Regression With L1 And L2 Regularization")
v = len(data.columns)
X = data.iloc[:, 1:v - 1].values
Y = data.iloc[:, v - 1].values

# Splitting Data_Sets
val = len(X) // 3
X_train = X[val:]
X_test = X[: val]
Y_train = Y[val:]
Y_test = Y[: val]

model = Logistic_Regression(1000, 0.021, X_train, Y_train, 0.5)
model.fit()
model.cost_plot()
predictions = model.predict(X_test)
print("Predictions")
print(predictions)
print("Original Values")
print(Y_test)
print("Accuracy Score : ", Regression.accuracy_metric(Y_test, predictions))
