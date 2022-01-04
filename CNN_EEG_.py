#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries

import os
from os import listdir
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import MaxPooling2D, Convolution2D
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
import scikitplot as skplt


# In[2]:


# input image dimensions

img_rows, img_cols = 4, 4


# In[3]:


# all subjects csv files folder

DATA_FOLDER = "output"

# subject wise performance result csv

SUBJECT_WISE_PERFORMANCE_METRIC_CSV = "cnn_subject_performance_metric.csv"


# In[4]:


# list all the subject wise csv files

def find_csv_filenames(path_to_dir, suffix=".csv"):
    filenames = listdir(path_to_dir)
    return [os.path.join(path_to_dir, filename) for filename in filenames if filename.endswith(suffix)]


# In[5]:


# load dataframe from csv

def load_data(filename):
    # read csv file
    df = pd.read_csv(filename)
    return df


# In[6]:


def class_scores(y_true, y_pred, reference):
    
    """Function which takes two lists and a reference indicating which class
    to calculate the TP, FP, and FN for."""
    
    # .................................................................
    Y_true = set([i for (i, v) in enumerate(y_true) if v == reference])
    # print("Y_true:{}".format(Y_true))
    Y_pred = set([i for (i, v) in enumerate(y_pred) if v == reference])
    # print("Y_pred:{}".format(Y_pred))
    TP = len(Y_true.intersection(Y_pred))
    # print(TP)
    FP = len(Y_pred - Y_true)
    FN = len(Y_true - Y_pred)
    return TP, FP, FN


# In[7]:


def f_beta_score(precision, recall, beta=1):
    """A function which takes the precision and recall of some model, and a value for beta,
    and returns the f_beta-score"""
    #.......................................................................
    return (1+beta**2) * precision * recall / (beta**2 * precision + recall)


# In[8]:


def f_score(precision, recall):
    return f_beta_score(precision, recall, beta=1)


# In[9]:


# train model function

def train_model(filename):
    extracted_filename = os.path.basename(filename)
    
    # split the filename by first underscore
    lhs, rhs = extracted_filename.split("_", 1)

    df = load_data(filename)

    # pop out label from dataframe
    y = df.pop("class")
    X = df

    # data split into training and testing dataset
    x_train, x_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, shuffle=True)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Reshape and normalize training data
    x_train = x_train.reshape(x_train.shape[0], 1, 4, 4).astype('float32')
    x_train = x_train / 255.0

    # Reshape and normalize test data
    x_test = x_test.reshape(x_test.shape[0], 1, 4, 4).astype('float32')
    x_test = x_test / 255.0

    # input shape of the model
    input_shape = (1, img_rows, img_cols)

    # convert labels into categories
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # some parameters
    batch_size = 264
    num_classes = 2
    epochs = 1000

    # the neural architecture
    model = Sequential()
    model.add(Convolution2D(30, 5, 5, padding='same', input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # train the model
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        shuffle=True)

    # history plot for accuracy and loss
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy of {}'.format(lhs))
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('graphs/Model Accuracy of {}.png'.format(lhs))
    # clean the current fig for save plotting
    plt.clf()
    # plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss of {}'.format(lhs))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('graphs/Model Loss of {}.png'.format(lhs))
    # clean the current fig for save plotting
    plt.clf()
    # plt.show()


    # convert them to single-digit ones
    predictions = model.predict_classes(x_test)

    #convert them too to single-digit ones
    y_test = np.argmax(y_test, axis=1)

    report = classification_report(y_test, predictions,  output_dict=True)
    print(report)

    print(history.history['accuracy'][-1], history.history['val_accuracy'][-1], history.history['loss'][-1],           history.history['val_loss'][-1])

    return lhs, history.history['accuracy'][-1], history.history['val_accuracy'][-1], history.history['loss'][-1],            history.history['val_loss'][-1], report


# In[10]:


if __name__ == '__main__':
    subject_filenames = find_csv_filenames(DATA_FOLDER)

    # new dataframe for performance metrics
    merged_subjects_df = pd.DataFrame()

    for name in subject_filenames:
        subject_id, acc, val_acc, loss, val_loss, report = train_model(name)
        subject_dic = {"subject_id": subject_id, "train_acc": acc, "test_acc": val_acc, "classification report":report}

        merged_subjects_df = merged_subjects_df.append(subject_dic, ignore_index=True)

    # write into csv file
    merged_subjects_df.to_csv(SUBJECT_WISE_PERFORMANCE_METRIC_CSV, index=True)


# In[ ]:





# In[ ]:





# In[ ]:




