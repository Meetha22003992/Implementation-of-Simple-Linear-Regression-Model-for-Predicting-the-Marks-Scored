# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import panda as pd 

2. Import numpy as np 

3. Copy the path of the file and link to the program

4. Using MSE, MAE and RMSE formule print the values of the predictions

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Meetha Prabhu
RegisterNumber:  212222240065

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv('/content/student_scores.csv')
print("Student Scores:")
df.head()

print('df.tail:')
df.tail()

print("X values:")
X=df.iloc[:,:-1].values
X

print("Y values:")
Y=df.iloc[:,1].values
Y

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

Y_pred

 Y_test
 
 plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
print('Training Set:')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores(Test set)")
print("Test set:")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE= ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE= ",rmse)
*/
```

## Output:
![image](https://github.com/Meetha22003992/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119401038/e717a249-bf18-43ec-a3b8-0ef4a9c713f2)

![image](https://user-images.githubusercontent.com/119401038/229291125-e50d2997-2349-43b2-8201-0fc8d246d8a2.png)

![image](https://user-images.githubusercontent.com/119401038/229291153-cee7acc9-f0f4-485e-ba42-b03770e5fae5.png)

![image](https://user-images.githubusercontent.com/119401038/229291171-10496903-fbf8-4cd5-8ac2-df269741571b.png)

![image](https://github.com/Meetha22003992/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119401038/9b489838-ec7c-4e37-a16d-d0415462e402)

![image](https://github.com/Meetha22003992/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119401038/2351fab4-56dd-425c-9047-c5d030fd6fbb)

![image](https://user-images.githubusercontent.com/119401038/229291213-535f7864-a346-4e9b-b998-a8a7b144800d.png)

![image](https://user-images.githubusercontent.com/119401038/229291222-71154c15-007a-4520-b2ae-5e34db68ef0a.png)

![image](https://github.com/Meetha22003992/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119401038/c7435e0e-533d-4842-af3e-29950b3bf091)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
