import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from linear_regression_model import Linear_Regression
from sklearn.metrics import mean_squared_error
import numpy as np

# Data Pre-Processiing
dataset = pd.read_csv('./salary_data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values

# Splitting data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y , test_size=0.35, random_state=1)

model = Linear_Regression(learning_rate = 0.02, no_of_epochs = 1000)
model.fit(X_train, Y_train)

Y_test_pred = model.predict(X_test)
plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_test, Y_test_pred, color='green')
plt.show()

