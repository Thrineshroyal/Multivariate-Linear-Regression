# Implementation of Multivariate Linear Regression
## Aim
To write a python program to implement multivariate linear regression and predict the output.
## Equipment’s required:
1.	Hardware – PCs
2.	Anaconda – Python 3.7 Installation / Moodle-Code Runner
## Algorithm and Steps:
### Import Libraries:

Import the necessary libraries, such as pandas for data manipulation and scikit-learn for machine learning.
### Read Data:

Read the dataset from a CSV file using the read_csv function from pandas.
### Define Features and Target:

Select the features (independent variables) and the target variable (dependent variable) from the dataset. In this case, 'Weight' and 'Volume' are the features, and 'CO2' is the target variable.
### Create a Linear Regression Model:

Initialize a linear regression model using the LinearRegression class from scikit-learn.
### Split Data (if necessary):

Split the dataset into training and testing sets if you want to evaluate the model's performance on unseen data.
### Train the Model:

Use the fit method to train the linear regression model on the training data, passing the features (x) and target (y) as arguments.
### Get Coefficients and Intercept:

Retrieve the coefficients and intercept of the trained model using regr.coef_ and regr.intercept_.
### Make Predictions:

Use the trained model to make predictions. In your example, you predict the CO2 emissions for a specific combination of weight and volume.
### Print Results:

Print the coefficients, intercept, and the predicted CO2 value.
## Program:
```
'''
Program for multivariate linear regression and to predict a unknown variable using the given dataset.
Developed By: Thrinesh Royal
Register Number: 212223230226
'''
import pandas as pd
from sklearn import linear_model
data=pd.read_csv("cars (1) (1).csv")
x=data[['Weight','Volume']]
y=data[['CO2']]
regr=linear_model.LinearRegression()
regr.fit(x,y)
print("Coefficients:",regr.coef_)
print("Intercept:",regr.intercept_)
predictCO2=regr.predict([[3300,1300]])
print("Predicted CO2 for the corresponding weight and volume",predictCO2)
```
## Output:
![image](https://github.com/SANTHAN-2006/Multivariate-Linear-Regression/assets/80164014/9a29e2bd-868a-41aa-bfae-4c83a388ab6f)

## Result
Thus the multivariate linear regression is implemented and predicted the output using python program.
