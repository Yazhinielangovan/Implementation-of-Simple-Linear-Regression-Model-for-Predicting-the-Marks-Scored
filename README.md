# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: YAZHINI.E
RegisterNumber: 2305002028 
*/
```
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```
## Output:
df.head()

![image](https://github.com/Yazhinielangovan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/155508323/e45fbc01-3f6a-41db-b335-8e76b43cf15a)

df.tail()

![image](https://github.com/Yazhinielangovan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/155508323/6acd01c9-2d71-4979-b52b-1ab1db5a7772)

Array value of X


![image](https://github.com/Yazhinielangovan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/155508323/997fc38b-d113-4146-a6cf-4a178a25794b)

Array value of Y


![image](https://github.com/Yazhinielangovan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/155508323/cfd573f6-128d-4430-88e3-43e806351cbb)

Values of Y prediction


![image](https://github.com/Yazhinielangovan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/155508323/44f7f4c4-219f-43c9-b3b9-33c5f781b9b2)

Array values of Y test


![image](https://github.com/Yazhinielangovan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/155508323/3387c221-2e65-49cb-b0a1-7cdbc3aa8a8b)

Training Set Graph


![image](https://github.com/Yazhinielangovan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/155508323/b901ecc9-91ad-41b9-add5-c08642e44c45)

Test Set Graph

![image](https://github.com/Yazhinielangovan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/155508323/25234cfc-15fe-491f-b490-43e083ed9344)

Values of MSE, MAE and RMSE


![image](https://github.com/Yazhinielangovan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/155508323/58e4ad88-38da-4485-8b31-dbd3497b5e5b)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
