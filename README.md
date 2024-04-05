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

![image](https://github.com/HIRU-VIRU/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145972122/5968514c-05e2-4d71-b132-b0acea5efd47)

![image](https://github.com/HIRU-VIRU/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145972122/ea9cb89a-f4b8-473d-84b8-1f92b2ee64a2)


Compare Dataset

![image](https://github.com/HIRU-VIRU/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145972122/9d7409fe-cb21-4727-9bd1-365c85ab9f1a)


Predicted Value

![image](https://github.com/HIRU-VIRU/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145972122/44b39961-15f3-4ef1-b64f-01bd7c4fff12)



Graph For Training Set
![image](https://github.com/HIRU-VIRU/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145972122/be5ed2ff-790c-4d0d-84c0-a230f9f4d2df)



Graph For Testing Set

![image](https://github.com/HIRU-VIRU/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145972122/3871af46-764a-496a-ad7d-39a81d931dee)


Error

![image](https://github.com/HIRU-VIRU/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145972122/b2a0e793-2aef-4ae5-ad28-79ec7f2399be)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
