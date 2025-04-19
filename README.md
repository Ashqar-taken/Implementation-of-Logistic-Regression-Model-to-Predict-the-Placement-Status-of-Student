# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. Display the results.

## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Ashqar Ahamed S.T
RegisterNumber:  212224240018
```
```
import pandas as pd
data=pd.read_csv(r'C:\College\SEM 2\Machine Learning\Exp5\Placement_Data.csv')
print("Data:")
print(data.head())

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
print("Data1: ")
print(data1.head())

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])     
print("data after labelencoder:")  
print(data1) 

x=data1.iloc[:,:-1]
print("X values:")
print(x)

y=data1["status"]
print("Y values: ")
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print("Predicted values:")
print(y_pred)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy: ")
print(accuracy)

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
print("Confusion Matrix: ")
print(confusion)

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print("The classification report: ")
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Data Values:
![Data values](https://github.com/user-attachments/assets/a3699a1b-4e01-4cc9-be78-3c4b56540b46)

## Data Values after labelencoder:
![data after labelencoder](https://github.com/user-attachments/assets/4c6e26e7-03d7-416e-ac47-00fe7b5b04f9)

## X values:
![X  values](https://github.com/user-attachments/assets/21e8268e-f236-4a83-86ed-d3cf2ccdf067)

## Y values:
![Y values](https://github.com/user-attachments/assets/e88b74b4-ade1-482a-90ef-5f77d180d939)

## Predicted Values:
![Predicted values](https://github.com/user-attachments/assets/c51acba5-8e26-40c7-8443-0637ece2190b)

## Accuracy:
![Accuracy](https://github.com/user-attachments/assets/49239098-655c-4c26-a142-5479844c90ab)

## Confusion matrix:
![confusion matrix](https://github.com/user-attachments/assets/08e55e71-c729-4f27-99de-c3698b2b7b5c)

## Classification report:
![classification report](https://github.com/user-attachments/assets/1b56ff62-62f1-4b5e-bc86-fd1228953f42)

## Prediciton:
![prediction ](https://github.com/user-attachments/assets/e7ade7e2-e672-4945-9ca4-37873051d5a0)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
