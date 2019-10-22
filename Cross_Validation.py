from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from scipy.stats import zscore
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow.keras.layers import Dense, Activation
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold

# LER OS DADOS E SEPARA-LOS
dataset = pd.read_csv('dadaSetMDD.csv', header=0)

x = dataset.iloc[:,1:54676]
y = dataset.iloc[:,0].values

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# Cross-Validate
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
oos_y = []
oos_pred = []
models = []
cvscores = []
history = []
fold = 0
for train, test in kf.split(x, y):
    fold+=1
    print(f"Fold #{fold}")
    x_train = x.loc[train]
    y_train = y.loc[train]
    x_test = x.loc[test]
    y_test = y.loc[test]
    
    # Salvar best results
    filepath= str(fold)+"-val-loss3.hdf5"
    filepath2= str(fold)+"-val-acc3.hdf5"
    filepath3= str(fold)+"-loss3.hdf5"
    filepath4= str(fold)+"-acc3.hdf5"
    mcp_save = ModelCheckpoint(filepath, save_best_only=True, verbose=0, monitor='val_loss', mode='min')
    acura_save = ModelCheckpoint(filepath2, save_best_only=True, verbose=0, monitor='val_accuracy', mode='max')
    mcp_savew = ModelCheckpoint(filepath3, save_best_only=True, verbose=0, monitor='loss', mode='min')
    acura_savew = ModelCheckpoint(filepath4, save_best_only=True, verbose=0, monitor='accuracy', mode='max')
    
    model = tf.keras.Sequential()
    model.add(Dense(100,kernel_initializer='random_normal', input_dim=54675, activation='relu'))
    model.add(Dense(100,kernel_initializer='random_normal', activation='relu'))
    model.add(Dense(1,kernel_initializer='random_normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    history.append(model.fit(x_train, y_train, callbacks=[mcp_save, acura_save, mcp_savew, acura_savew],validation_data=(x_test,y_test),verbose=2, batch_size=20, epochs=500))
    
    pred = model.predict(x_test)
    
    oos_y.append(y_test)
    oos_pred.append(pred)
    models.append(model)
    # Measure this fold's RMSE
    score = np.sqrt(metrics.mean_squared_error(pred,y_test))
    print(f"Fold score (RMSE): {score}")

# Build the oos prediction list and calculate the error.
oos_y = np.concatenate(oos_y)
oos_pred = np.concatenate(oos_pred)
score = np.sqrt(metrics.mean_squared_error(oos_pred,oos_y))
print(f"Final, out of sample score (RMSE): {score}")

# best result histoy per fold
for result in history:
    validation_acc = np.amax(result.history['val_accuracy'])
    print(validation_acc)

