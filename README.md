# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import the needed packages. 
2. Assigning hours to x and scores to y.
3. Plot the scatter plot.
4. Use mse,rmse,mae formula to find the values.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: HIRUTHIK SUDHAKAR
RegisterNumber: 212223240054
*/
```
```python
# IMPORT REQUIRED PACKAGE
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
dataset=pd.read_csv('student_scores.csv')
print(dataset)
# READ CSV FILES
dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())
# COMPARE DATASET
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,1].values
print(y)
# PRINT PREDICTED VALUE
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)
# GRAPH PLOT FOR TRAINING SET
plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
# GRAPH PLOT FOR TESTING SET
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
# PRINT THE ERROR
mse=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse)

```

## Output:

To Read Head and Tail Files

![image](https://github.com/HIRU-VIRU/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145972122/1608a2c9-31cd-4251-9c38-008e7c2ee39f)


Compare Dataset

![image](https://github.com/HIRU-VIRU/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145972122/713447d8-ba83-4ce7-9882-168ed768c2b2)


Predicted Value

![image](https://github.com/HIRU-VIRU/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145972122/4c86c629-ab8b-42a9-923d-d2b60a778c12)


Graph For Training Set

![image](https://github.com/HIRU-VIRU/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145972122/359a3ab5-f963-4083-aa25-5e9ab7002b2e)


Graph For Testing Set

![image](https://github.com/HIRU-VIRU/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145972122/a68f0377-2dc5-4723-bf8c-fa9a22fa9687)


Error

![image](https://github.com/HIRU-VIRU/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145972122/4daf37cf-a940-482c-ab98-6ebedd8f4aab)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
