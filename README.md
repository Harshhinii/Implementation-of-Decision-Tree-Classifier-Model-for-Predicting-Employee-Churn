# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import and Load Data: Import necessary libraries (pandas, sklearn) and load the employee churn dataset.

2.Preprocess Data: Handle missing values, encode categorical variables, and separate features (X) and target (y).

3.Split Data: Divide the dataset into training and testing sets using train_test_split().

4.Train Model: Create and train a DecisionTreeClassifier on the training data.

5.Predict and Evaluate: Use the model to predict churn on test data and evaluate performance using accuracy and confusion matrix.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: HARSHINI R
RegisterNumber: 212223220033 
*/
```
```
import pandas as pd 
import numpy as np
df=pd.read_csv("Employee.csv")
print(df.head())
```
<img width="898" height="449" alt="image" src="https://github.com/user-attachments/assets/4e2be06b-9cbf-4744-a54f-305b114a2f61" />

```
df.info()
```

<img width="581" height="372" alt="image" src="https://github.com/user-attachments/assets/deb54182-f4a6-4aae-a9ba-69d77a7eab46" />

```
df.isnull().sum()
```

<img width="318" height="245" alt="image" src="https://github.com/user-attachments/assets/232de253-a0a2-47ef-99d7-a24644953788" />

```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["salary"]=le.fit_transform(df["salary"])
df.head()
```

<img width="1396" height="250" alt="image" src="https://github.com/user-attachments/assets/6bee03ac-d01e-4303-b7aa-ccedcf08422b" />

```
df["left"].value_counts()
```
<img width="486" height="110" alt="image" src="https://github.com/user-attachments/assets/64ca49b4-01fc-4fe9-a29b-30a8e448721c" />

```
x=df[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
```

<img width="1402" height="235" alt="image" src="https://github.com/user-attachments/assets/2809e657-d92e-486e-a46c-004e6d0e24bf" />

```
y=df["left"]
y.head()
```
<img width="501" height="177" alt="image" src="https://github.com/user-attachments/assets/01cb4f9e-cadb-407d-9f5d-bdb87493dc85" />

```
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(X_train,Y_train)
y_pred=dt.predict(X_test)
print("Name: HARSHINI R ")
print("RegNo: 212223220033")
print(y_pred)
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(Y_test,y_pred)
cm=confusion_matrix(Y_test,y_pred)
cr=classification_report(Y_test,y_pred)
print("Accuracy:",accuracy)
print("Confusion Matrix:")
print(cm)
print("Classification Report:")
print(cr)
dt.predict(pd.DataFrame([[0.6,0.9,8,292,6,0,1,2]],columns=["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]))
```
## Output:

<img width="1920" height="1020" alt="image" src="https://github.com/user-attachments/assets/449f19b4-2631-4013-94f2-4a206cb2aacb" />




## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
