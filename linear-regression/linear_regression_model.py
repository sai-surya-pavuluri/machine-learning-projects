import numpy as np

# Building Linear Regression model
class Linear_Regression():

    def __init__(self, learning_rate, no_of_epochs):
        
        self.learning_rate = learning_rate
        self.no_of_epochs = no_of_epochs

    def fit(self, X, Y):
        self.m, self.n = X.shape
        self.w = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        for epoch in range(self.no_of_epochs):
           self.update_weights() 

    def update_weights(self):

        Y_prediction = self.predict(self.X)

        dw = -(2*(self.X.T).dot(self.Y - Y_prediction))/self.m
        db = -2 * np.sum(self.Y - Y_prediction)/self.m

        self.w = self.w - self.learning_rate*dw
        self.b = self.b - self.learning_rate*db

    def predict(self, X):
        return X.dot(self.w) + self.b
