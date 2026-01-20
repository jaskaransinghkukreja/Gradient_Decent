class Gradient_Decent_Scratch:       # Gradient Decent Code from Scratch using class     
    def __init__(self,m,b,lr,epochs):
        self.m=m
        self.b=b
        self.lr=lr
        self.epochs=epochs


    def fit(self,X,y):

        plt.scatter(X,y)
        pred_1=self.m * X +self.b
        plt.plot(X,pred_1,color="red",label="Assumed BFL")
        # plt.show()
        for i in range(self.epochs):
            error_b=0
            error_m=0
            for j in range(len(X)):
                error_b=error_b + (y[j] - (self.m * X[j]) - self.b)
                error_m=error_m + (((y[j] - (self.m * X[j]) - self.b))* X[j])

            error_b=error_b * (-2)
            error_m=error_m * (-2)

            self.b = self.b - (self.lr * error_b)
            self.m=self.m - (self.lr * error_m)
            
            pred_2=self.m * X +self.b
            plt.plot(X,pred_2,color="green")
        
        print(f'Intercept of the Best Fit Line using our class:{self.b}')
        print(f'Slope of the Best Fit Line using our class:{self.m}')
        
        pred_3=self.m * X +self.b
        plt.plot(X,pred_3,color="purple",label="Class BFT")
        





#Training model using Scikit learn and calling Class(Gradient_Decent_Scratch)

import pandas  as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

lr= LinearRegression()

X,y=make_regression(n_samples=100,n_features=1,n_informative=1,n_targets=1,noise=52,random_state=42)

lr.fit(X,y)

b=float(input("Enter the value of the Intercept:"))
m=float(input("Enter the value of the Slope:"))
learn_rate=float(input("Enter the Learning Rate:"))
epochs=int(input("Enter the Epochs(number of iterations):"))

model=Gradient_Decent_Scratch(m,b,learn_rate,epochs)
model.fit(X,y)

y_pred=lr.predict(X)
plt.plot(X,y_pred,color="magenta",label="Original BFL")
plt.legend()
plt.show()
print(f'Intercept(Scikit-learn model): {lr.intercept_}\nCoefficient(Scikit-learn model): {lr.coef_}')
