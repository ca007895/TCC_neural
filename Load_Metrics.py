from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from keras.utils import to_categorical
from sklearn import preprocessing

# LER OS DADOS E SEPARA-LOS
dataset = pd.read_csv('dadaSetMDD.csv') 
x = dataset.iloc[:,1:54676]
y = dataset.iloc[:,0]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = Sequential()
model.add(Dense(100,kernel_initializer='random_normal', input_dim=54675, activation='relu'))
model.add(Dense(50,kernel_initializer='random_normal', activation='relu'))
model.add(Dense(1,kernel_initializer='random_normal', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


model.load_weights("3-val-loss2.hdf5") # load arquivo salvo de metrica

# printar results
_, accuracy = model.evaluate(x, y)
print('Modelo Completo')
print('Accuracy: %.2f' % (accuracy*100)) 

y_pred=model.predict(x)
y_pred =(y_pred>0.5)
cm = confusion_matrix(y, y_pred)
print(cm)

model.summary()