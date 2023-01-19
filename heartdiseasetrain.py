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
print(x.shape,x_train.shape,x_test.shape)
model = LogisticRegression()
#training
model.fit(x_train,y_train)