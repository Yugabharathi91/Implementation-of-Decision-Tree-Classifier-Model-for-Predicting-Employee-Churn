# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import pandas module and import the required data set.
2. Find the null values and count them.
3. Count number of left values.
4. From sklearn import LabelEncoder to convert string values to numerical values.
5. From sklearn.model_selection import train_test_split.
6. Assign the train dataset and test dataset.
7. From sklearn.tree import DecisionTreeClassifier.
8. Use criteria as entropy.
9. From sklearn import metrics.
10. Find the accuracy of our model and predict the require values.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: GOKUL S
RegisterNumber:212224230075  
*/
```
```
import pandas as pd
data=pd.read_csv("/content/Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```

## Output:
```
data.head()
```
<img width="1234" height="219" alt="image" src="https://github.com/user-attachments/assets/83d40b87-8b66-474c-adc9-6b1ac0080737" />

```
data.info()
```
<img width="712" height="304" alt="image" src="https://github.com/user-attachments/assets/19823581-2c93-446f-85ac-a1b54ab9448b" />


```
data.isnull().sum()
```
<img width="334" height="400" alt="image" src="https://github.com/user-attachments/assets/721be95e-74a6-4233-ba2b-10edcd95582d" />

```
data value counts()
```
<img width="402" height="109" alt="image" src="https://github.com/user-attachments/assets/7e4a1326-fa88-491d-808c-8a5d954e42af" />

```
data.head() for salary
```
<img width="1229" height="212" alt="image" src="https://github.com/user-attachments/assets/771e01e1-b740-4e63-879f-13b0c6a08450" />

```
x.head()
```
<img width="1086" height="202" alt="image" src="https://github.com/user-attachments/assets/4d32c4ad-eb32-464f-a23a-f693d7d6f8ac" />

```
accuracy
```
<img width="245" height="33" alt="image" src="https://github.com/user-attachments/assets/018fa34b-b7e4-4d8f-b8ed-6d72f840df69" />

```
Data Prediction
```
<img width="1661" height="113" alt="image" src="https://github.com/user-attachments/assets/0b195790-70df-4bda-80ed-e5b67800817d" />


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
