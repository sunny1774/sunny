# sunny
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
%matplotlib inline
cars = pd.read_csv(r"C:\Users\RCPIT\Downloads\cars (2).csv")
print(cars)
cars.head()
cars.tail()
cars.describe()
cars.info()
X = cars[['Weight', 'Volume']]
y = cars['CO2']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=0)
regressor =LinearRegression(fit_intercept=True)
regressor.fit(X_train,y_train)
print('Linear Model Coeff (m) =' , regressor.coef_)
print('Linear Model Coeff (b) =' , regressor.intercept_)
y_predict=regressor.predict(X_test)
print(y_predict)
print('---------[700,900]---------')
wg = 700
vol = 900
co2 = regressor.predict([[wg,vol]])
print(co2)
print('--------[1100,1500]----------')
wg = 1100
vol = 1500
co2 = regressor.predict([[wg,vol]])
print(co2)
print('--------[1500,2500]----------')
wg = 1500
vol = 2500
co2 = regressor.predict([[wg,vol]])
print(co2)
