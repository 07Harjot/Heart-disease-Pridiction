import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#dataproccessing
heartdata = pd.read_csv('C:\\Users\\acer\\Desktop\\Heart disease Pridiction\\heart_disease_data.csv')

#print(heartdata)
#print(heartdata.info())
#print(heartdata.describe())
#print(heartdata['target'].value_counts())

#1-->Defect
#0-->healthy

x = heartdata.drop(columns='target',axis=1)
y = heartdata['target']

#print(y)

#spliting data into training and test data
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)
#print(x.shape,x_train.shape,x_test.shape)
model = LogisticRegression()
#training
model.fit(x_train,y_train)

#accuraracy on training data
X_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(X_train_prediction,y_train)
print('accuracy on training data :',training_data_accuracy)

#accuraracy on test data
X_test_prediction = model.predict(x_test)
testing_data_accuracy = accuracy_score(X_test_prediction,y_test)
print('accuracy on training data :',testing_data_accuracy)
