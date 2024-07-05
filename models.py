from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import pandas as pd
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import PredefinedSplit
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.metrics import accuracy_score,f1_score,recall_score, roc_auc_score
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import PredefinedSplit

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.svm import SVC

from xgboost import XGBRegressor, XGBClassifier

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D, Conv2D
from keras.layers.convolutional import MaxPooling1D, MaxPooling2D
from keras.layers import LSTM, SimpleRNN,GRU
from keras.layers import Dropout

import tensorflow.keras.backend as K
import tensorflow as tf

import warnings
warnings.filterwarnings('ignore')

#import by Anh
import os
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Input, Embedding, Dense, MaxPool2D, concatenate
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K

from keras import initializers, regularizers, constraints, optimizers, layers
from keras.layers import concatenate
from keras.callbacks import *
def rmse(y_test, pred):
  return np.sqrt(mse(y_test,pred))

def add_lag_and_split_data_temp():

  path = os.getcwd()
  print(path)
  dataset = pd.read_csv('/content/drive/My Drive/datasets/sensor_raw.csv',index_col=[0])
  dataset= dataset.reset_index()
  dataset = dataset[['GyroX', 'GyroY', 'GyroZ', 'AccX','AccY','AccZ','Target(Class)']]
  cols = [col for col in dataset.columns if col != 'Target(Class)']


  for i, target_cls in enumerate(dataset['Target(Class)'].unique()):
    new_dataset = dataset[dataset['Target(Class)'] == target_cls].copy()


    trn_size = int(0.6 * new_dataset.shape[0])
    val_size = int(0.8 * new_dataset.shape[0])

    new_dataset_trn = new_dataset[:trn_size].copy()
    new_dataset_val = new_dataset[trn_size:val_size].copy()
    new_dataset_tst = new_dataset[val_size:].copy()



    if i == 0:
      data_trn = new_dataset_trn
      data_val = new_dataset_val
      data_tst = new_dataset_tst
      continue


    data_trn = data_trn.append(new_dataset_trn)
    data_val = data_val.append(new_dataset_val)
    data_tst = data_tst.append(new_dataset_tst)


  x_trn = data_trn.drop(['Target(Class)'], axis=1)
  x_val = data_val.drop(['Target(Class)'], axis=1)
  x_tst = data_tst.drop(['Target(Class)'], axis=1)

  y_trn = data_trn['Target(Class)']
  y_val = data_val['Target(Class)']
  y_tst = data_tst['Target(Class)']

  return x_trn, x_val, x_tst, y_trn, y_val, y_tst





x_trn, x_val, x_tst, y_trn, y_val, y_tst = add_lag_and_split_data_temp()

y_trn.unique()

def get_driving_pred_data_nn(window = 4):

    x_trn, x_val, x_tst, y_trn, y_val, y_tst = add_lag_and_split_data_temp()

    #Scailing data
    scalerX = MinMaxScaler().fit(x_trn)
    x_trn = scalerX.transform(x_trn)
    x_val = scalerX.transform(x_val)
    x_tst = scalerX.transform(x_tst)


    y_trn = y_trn.to_numpy()
    y_val = y_val.to_numpy()
    y_tst = y_tst.to_numpy()

    #Store the number of driver behaviours' features
    num_features = x_trn.shape[1]


    samples_train = x_trn.shape[0] - window
    x_trn_reshaped = np.zeros((samples_train, window, num_features)) # Initialize the required shape with an 'empty' zeros array.
    y_trn_reshaped = np.zeros((samples_train)).astype(np.int)
    for i in range(samples_train):
        y_position = i + window
        x_trn_reshaped[i] = x_trn[i:y_position]
        y_trn_reshaped[i] = y_trn[y_position]#[0]

    samples_val = x_val.shape[0] - window
    x_val_reshaped = np.zeros((samples_val, window, num_features))
    y_val_reshaped = np.zeros((samples_val)).astype(np.int)
    for i in range(samples_val):
        y_position = i + window
        x_val_reshaped[i] = x_val[i:y_position]
        y_val_reshaped[i] = y_val[y_position]#[0]


    samples_test = x_tst.shape[0] - window
    x_tst_reshaped = np.zeros((samples_test, window, num_features))
    y_tst_reshaped = np.zeros((samples_test)).astype(np.int)
    for i in range(samples_test):
        y_position = i + window
        x_tst_reshaped[i] = x_tst[i:y_position]
        y_tst_reshaped[i] = y_tst[y_position]#[0]

    x_trn, y_trn, x_val, y_val, x_tst, y_tst = x_trn_reshaped, y_trn_reshaped, x_val_reshaped, y_val_reshaped, x_tst_reshaped, y_tst_reshaped


    return x_trn, y_trn, x_val, y_val, x_tst, y_tst

x_trn, y_trn, x_val, y_val, x_tst, y_tst= get_driving_pred_data_nn()

def evaluate_classification_rnn_model(clf_nn_model, x_trn, y_trn, x_val, y_val, x_tst, y_tst):

    from matplotlib import pyplot as plt
    #Training the Regression models on the dataset
    start = time.time()
    history= clf_nn_model.fit(x_trn, y_trn, epochs=200, batch_size=32, validation_data = (x_val, y_val))
    end = time.time()

    #Predicting the Test set results
    y_tst_pred = np.argmax(clf_nn_model.predict(x_tst), axis=-1)

    #Predicting the Test set results
    accuracy = accuracy_score(y_tst, y_tst_pred)
    f1 = f1_score(y_tst, y_tst_pred, average='macro')
    training_time = end - start

    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.legend()

    return accuracy, training_time,f1

def make_rnn_model_classification(rnn_input_shape, n = 32):
    K.clear_session()
    model = Sequential()

    ## Adding the first LSTM layer and some Dropout regularisation
    model.add(LSTM(units = n, return_sequences = True, input_shape = rnn_input_shape))
    model.add(Dropout(0.2))

    ## Adding a second LSTM layer and some Dropout regularisation
    model.add(LSTM(units = n , return_sequences = True))
    model.add(Dropout(0.2))

    ## Adding a third LSTM layer and some Dropout regularisation
    model.add(LSTM(units = n, return_sequences = True))
    model.add(Dropout(0.2))

    ## Adding the output layer
    model.add(Flatten())
    model.add(Dense(units = 5, activation='sigmoid'))
    print(model.summary())

    # compile
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    return model





rnn_input_shape = (x_trn.shape[1], x_trn.shape[2])

reg_nn_model = make_rnn_model_classification(rnn_input_shape)

accuracy, training_time,f1 = evaluate_classification_rnn_model(reg_nn_model, x_trn, y_trn, x_val, y_val, x_tst, y_tst)
print('Accuracy : ', accuracy,'F1 : ',f1, ' Time :', training_time) #need to save in a file

LSTM :: Accuracy :  0.5900621118012422 F1 :  0.5990322889719268  Time : 30.66784954071045
GRU  :: Accuracy :  0.6521739130434783 F1 :  0.6451267854318837  Time : 34.435810565948486
RNN  :: Accuracy :  0.577639751552795 F1 :  0.5858813732438967  Time : 42.73585391044617
CNN  :: Accuracy :  0.6708074534161491 F1 :  0.6749502046566415  Time : 24.04517126083374



"""# CNN Model"""



def add_lag_and_split_data_temp():

  dataset = pd.read_csv('/content/drive/My Drive/Dataset/RasPi/sensor_raw.csv',index_col=[0])
  dataset= dataset.reset_index()
  dataset = dataset[['GyroX', 'GyroY', 'GyroZ', 'AccX','AccY','AccZ','Target(Class)']]
  cols = [col for col in dataset.columns if col != 'Target(Class)']


  for i, target_cls in enumerate(dataset['Target(Class)'].unique()):
    new_dataset = dataset[dataset['Target(Class)'] == target_cls].copy()


    trn_size = int(0.6 * new_dataset.shape[0])
    val_size = int(0.8 * new_dataset.shape[0])

    new_dataset_trn = new_dataset[:trn_size].copy()
    new_dataset_val = new_dataset[trn_size:val_size].copy()
    new_dataset_tst = new_dataset[val_size:].copy()



    if i == 0:
      data_trn = new_dataset_trn
      data_val = new_dataset_val
      data_tst = new_dataset_tst
      continue


    data_trn = data_trn.append(new_dataset_trn)
    data_val = data_val.append(new_dataset_val)
    data_tst = data_tst.append(new_dataset_tst)


  x_trn = data_trn.drop(['Target(Class)'], axis=1)
  x_val = data_val.drop(['Target(Class)'], axis=1)
  x_tst = data_tst.drop(['Target(Class)'], axis=1)

  y_trn = data_trn['Target(Class)']
  y_val = data_val['Target(Class)']
  y_tst = data_tst['Target(Class)']

  return x_trn, x_val, x_tst, y_trn, y_val, y_tst

def get_driving_pred_data_nn(window = 14):



    x_trn, x_val, x_tst, y_trn, y_val, y_tst = add_lag_and_split_data_temp()


    scalerX = MinMaxScaler().fit(x_trn)
    x_trn = scalerX.transform(x_trn)
    x_val = scalerX.transform(x_val)
    x_tst = scalerX.transform(x_tst)


    y_trn = y_trn.to_numpy()
    y_val = y_val.to_numpy()
    y_tst = y_tst.to_numpy()

    #Store the number of driver behaviours' features
    num_features = x_trn.shape[1]


    samples_train = x_trn.shape[0] - window
    x_trn_reshaped = np.zeros((samples_train, window, num_features)) # Initialize the required shape with an 'empty' zeros array.
    y_trn_reshaped = np.zeros((samples_train)).astype(np.int)
    for i in range(samples_train):
        y_position = i + window
        x_trn_reshaped[i] = x_trn[i:y_position]
        y_trn_reshaped[i] = y_trn[y_position]#[0]

    samples_val = x_val.shape[0] - window
    x_val_reshaped = np.zeros((samples_val, window, num_features))
    y_val_reshaped = np.zeros((samples_val)).astype(np.int)
    for i in range(samples_val):
        y_position = i + window
        x_val_reshaped[i] = x_val[i:y_position]
        y_val_reshaped[i] = y_val[y_position]#[0]


    samples_test = x_tst.shape[0] - window
    x_tst_reshaped = np.zeros((samples_test, window, num_features))
    y_tst_reshaped = np.zeros((samples_test)).astype(np.int)
    for i in range(samples_test):
        y_position = i + window
        x_tst_reshaped[i] = x_tst[i:y_position]
        y_tst_reshaped[i] = y_tst[y_position]#[0]

    x_trn, y_trn, x_val, y_val, x_tst, y_tst = x_trn_reshaped, y_trn_reshaped, x_val_reshaped, y_val_reshaped, x_tst_reshaped, y_tst_reshaped


    return x_trn, y_trn, x_val, y_val, x_tst, y_tst



def evaluate_classification_cnn_model(clf_nn_model, x_trn, y_trn, x_val, y_val, x_tst, y_tst):


    x_trn = x_trn.reshape(x_trn.shape[0], x_trn.shape[1], x_trn.shape[2], 1)
    x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], 1)
    x_tst = x_tst.reshape(x_tst.shape[0], x_tst.shape[1], x_tst.shape[2], 1)


    #Training the Regression models on the dataset
    start = time.time()
    clf_nn_model.fit(x_trn, y_trn, epochs=100, batch_size=32, validation_data = (x_val, y_val))
    end = time.time()

    #Predicting the Test set results
    y_tst_pred = np.argmax(clf_nn_model.predict(x_tst), axis=-1)

    #Predicting the Test set results
    accuracy = accuracy_score(y_tst, y_tst_pred)
    f1 = f1_score(y_tst, y_tst_pred, average='macro')
    training_time = end - start

    return accuracy, training_time,f1

def make_cnn_model_classification(cnn_input_shape, n = 16):
    K.clear_session()
    model = Sequential()

    model.add(Conv2D(filters=n, kernel_size=(5, 5), input_shape=cnn_input_shape, padding='same'))
    model.add(Conv2D(filters=n, kernel_size=(5, 5), padding='same'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=n, kernel_size=(5, 5), padding='same'))
    model.add(Conv2D(filters=n, kernel_size=(5, 5), padding='same'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Flatten())
    model.add(Dense(units = 5))

    # compile
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    return model

cnn_input_shape = (x_trn.shape[1], x_trn.shape[2], 1)
clf_nn_model = make_cnn_model_classification(cnn_input_shape)

# این قسمت تغییر کرده است
accuracy, training_time,f1 = evaluate_classification_cnn_model(clf_nn_model, x_trn, y_trn, x_val, y_val, x_tst, y_tst)
print('Accuracy : ', accuracy,'F1 : ',f1, ' Time :', training_time) #need to save in a file

"""## TCN MODEL"""

def get_driving_pred_data_nn(window = 14):



    x_trn, x_val, x_tst, y_trn, y_val, y_tst = add_lag_and_split_data_temp()


    scalerX = MinMaxScaler().fit(x_trn)
    x_trn = scalerX.transform(x_trn)
    x_val = scalerX.transform(x_val)
    x_tst = scalerX.transform(x_tst)


    y_trn = y_trn.to_numpy()
    y_val = y_val.to_numpy()
    y_tst = y_tst.to_numpy()

    #Store the number of driver behaviours' features
    num_features = x_trn.shape[1]


    samples_train = x_trn.shape[0] - window
    x_trn_reshaped = np.zeros((samples_train, window, num_features)) # Initialize the required shape with an 'empty' zeros array.
    y_trn_reshaped = np.zeros((samples_train)).astype(np.int)
    for i in range(samples_train):
        y_position = i + window
        x_trn_reshaped[i] = x_trn[i:y_position]
        y_trn_reshaped[i] = y_trn[y_position]#[0]

    samples_val = x_val.shape[0] - window
    x_val_reshaped = np.zeros((samples_val, window, num_features))
    y_val_reshaped = np.zeros((samples_val)).astype(np.int)
    for i in range(samples_val):
        y_position = i + window
        x_val_reshaped[i] = x_val[i:y_position]
        y_val_reshaped[i] = y_val[y_position]#[0]


    samples_test = x_tst.shape[0] - window
    x_tst_reshaped = np.zeros((samples_test, window, num_features))
    y_tst_reshaped = np.zeros((samples_test)).astype(np.int)
    for i in range(samples_test):
        y_position = i + window
        x_tst_reshaped[i] = x_tst[i:y_position]
        y_tst_reshaped[i] = y_tst[y_position]#[0]

    x_trn, y_trn, x_val, y_val, x_tst, y_tst = x_trn_reshaped, y_trn_reshaped, x_val_reshaped, y_val_reshaped, x_tst_reshaped, y_tst_reshaped


    return x_trn, y_trn, x_val, y_val, x_tst, y_tst



def evaluate_classification_cnn_model(clf_nn_model, x_trn, y_trn, x_val, y_val, x_tst, y_tst):


    x_trn = x_trn.reshape(x_trn.shape[0], x_trn.shape[1], x_trn.shape[2], 1)
    x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], 1)
    x_tst = x_tst.reshape(x_tst.shape[0], x_tst.shape[1], x_tst.shape[2], 1)


    #Training the Regression models on the dataset
    start = time.time()
    clf_nn_model.fit(x_trn, y_trn, epochs=100, batch_size=32, validation_data = (x_val, y_val))
    end = time.time()

    #Predicting the Test set results
    y_tst_pred = np.argmax(clf_nn_model.predict(x_tst), axis=-1)

    #Predicting the Test set results
    accuracy = accuracy_score(y_tst, y_tst_pred)
    f1 = f1_score(y_tst, y_tst_pred, average='macro')
    training_time = end - start

    return accuracy, training_time,f1
