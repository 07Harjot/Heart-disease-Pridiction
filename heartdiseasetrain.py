import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#dataproccessing
heartdata = pd.read_csv('C:\\Users\\acer\\Desktop\\Heart disease Pridiction\\heart_disease_data.csv')

#print(heartdata)
#print(heartdata.info())