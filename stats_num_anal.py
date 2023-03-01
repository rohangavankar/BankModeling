import  pandas as pd
from sklearn import datasets, linear_model
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from ProjectBank import do_etl
# Importing the dataset
dataset = do_etl("/Users/rohangavankar/PycharmProjects/test_app/bank/bank.csv")
X = dataset.loc[:, ["age","marital_no","education_no","balance","poutcome_no"]].values
y = dataset.loc[:, ['y_no']].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
importance = regr.coef_
print('Coefficients: \n', importance)
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))

# Make predictions usving the testing set
y_pred = regr.predict(X_test)
print(len(X_test), ",", len(y_test), ",", len(y_pred))
# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print('Mean squared error: %.2f'
    %mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))
