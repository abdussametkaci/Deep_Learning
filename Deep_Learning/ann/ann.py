# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 19:41:35 2019

@author: Abdussamet
"""

#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2. Veri Onisleme

#2.1. Veri Yukleme
veriler = pd.read_csv('Churn_modelling.csv')

#veri on isleme
X = veriler.iloc[:, 3:13].values
Y = veriler.iloc[:, 13].values

#encoder:  Kategorik -> Numeric
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 1] = le.fit_transform(X[:, 1])

le2 = LabelEncoder()

X[:, 2] = le2.fit_transform(X[:, 2])

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features=[1])
X=ohe.fit_transform(X).toarray()
X = X[:, 1:]

#verilerin egitim ve test icin bolunmesi
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

#3 Yapay Sinir Ağı
import keras
from keras.models import Sequential
from keras.layers import Dense#katman

classifier = Sequential()#yapay sinir ağ için bir model
classifier.add(Dense(6, init = "uniform", activation="relu", input_dim = 11))
#eklediğimiz katmanda 6 tane gizli katmanlı nöron, 11 tane giriş katmanlı nörona sahip bir katman 
classifier.add(Dense(6, init = "uniform", activation="relu"))#ikinci bir gizli katman ekledik
#ikinci katmanın girişi olmadığı için input_dim i sildik

classifier.add(Dense(1, init = "uniform", activation="sigmoid"))#çıkış katmanı

classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
classifier.fit(X_train, y_train, epochs = 50)

y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)#0.5 in üstündeyse değer true döndürür aksi taktirde false
#müşterilerin bırakıp bıtrakmayacağını kesin dille söylemek için yaptık ama bunu yapmasak da oran olara elde edriz

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)
