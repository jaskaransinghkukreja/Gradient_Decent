class Gradient_Decent_Scratch:  
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
        
        print(self.b)
        print(self.m)
        
        pred_3=self.m * X +self.b
        plt.plot(X,pred_3,color="purple",label="Final BFL")
        plt.legend()
        plt.show()






import pandas  as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

lr= LinearRegression()

X,y=make_regression(n_samples=100,n_features=1,n_informative=1,n_targets=1,noise=52,random_state=42)

lr.fit(X,y)
print(f'Intercept : {lr.intercept_}\nCoefficient : {lr.coef_}')

model=Gradient_Decent_Scratch(-100,-10,0.001,10)
model.fit(X,y)