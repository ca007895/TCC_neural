# imports
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# read Data
dataset = pd.read_csv('dadaSetMDD.csv', header=None)
x = dataset.iloc[:,1:54676].values
y = dataset.iloc[:,0].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

def c_model(optimizer, n_neurons):
    model = Sequential()
    model.add(Dense(n_neurons,  input_dim=x.shape[1], activation='relu'))
    model.add(Dense(1, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=c_model, initial_epoch=0, verbose=1)
# define the grid search parameters
neurons=[8,16,32,64,128,256,512]
batch_sizes = [100, 150, 300, 500]
epochs = [10, 20, 50, 70, 1024]
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
#this does 3-fold classification. One can change k. 
param_grid = dict(n_neurons=neurons)
parameters = {'batch_size': batch_sizes, 'epochs': epochs ,'optimizer': optimizer,'n_neurons':neurons}
grid = GridSearchCV(estimator=model, param_grid=parameters, n_jobs=1)
grid_result = grid.fit(x_train, y_train)


# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))