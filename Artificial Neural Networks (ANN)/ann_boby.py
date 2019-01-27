#0 -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, 13].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]


from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size =0.2, random_state = 42)

from sklearn.preprocessing import StandardScaler
s = StandardScaler()
x_train = s.fit_transform(x_train)
x_test = s.fit_transform(x_test)

import keras

from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(units =6,kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

classifier.add(Dense(units =6,kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dense(units =1,kernel_initializer = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(x_train, y_train, batch_size = 10, nb_epoch = 100)


y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
