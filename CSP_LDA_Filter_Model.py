#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries

import os
from os import listdir
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import CSP


# In[2]:


# all subjects csv files folder

DATA_FOLDER = "output"

# subject wise performance result csv

SUBJECT_WISE_PERFORMANCE_METRIC_CSV_Filter = "csp_lda_subject_performance_metric_Filter.csv"


# In[3]:


# list all the subject wise csv files
def find_csv_filenames(path_to_dir, suffix=".csv"):
    filenames = listdir(path_to_dir)
    return [os.path.join(path_to_dir, filename) for filename in filenames if filename.endswith(suffix)]


# In[4]:


# load dataframe from csv
def load_data(filename):
    # read csv file
    df = pd.read_csv(filename)
    return df


# In[5]:


# Pre-process data with Lowerpass bank filter and model building

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from scipy.signal import butter, lfilter


class FilterBank(BaseEstimator, TransformerMixin):

# obtained from https://www.kaggle.com/eilbeigi/visual/data
# author: fornax, alexandre

    """Filterbank TransformerMixin.
    Return signal processed by a bank of butterworth filters.
    """

    def __init__(self, filters='LowpassBank'):
        """init."""
        if filters == 'LowpassBank':
            self.freqs_pairs = [[0.5], [1], [2], [3], [4], [5], [7], [9], [15],
                                [30]]
        else:
            self.freqs_pairs = filters
        self.filters = filters

    def fit(self, X, y=None):
        """Fit Method, Not used."""
        return self

    def transform(self, X, y=None):
        """Transform. Apply filters."""
        X_tot = None
        for freqs in self.freqs_pairs:
            if len(freqs) == 1:
                b, a = butter(5, freqs[0] / 250.0, btype='lowpass')
            else:
                if freqs[1] - freqs[0] < 3:
                    b, a = butter(3, np.array(freqs) / 250.0, btype='bandpass')
                else:
                    b, a = butter(5, np.array(freqs) / 250.0, btype='bandpass')
            X_filtered = lfilter(b, a, X, axis=0)
            X_tot = X_filtered if X_tot is None else np.c_[X_tot, X_filtered]

        return X_tot




# In[6]:


def preprocessData(data):
    """Preprocess data with filterbank."""
    fb = FilterBank()
    return fb.transform(data)


# In[7]:


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


# In[8]:


def f_beta_score(precision, recall, beta=1):
    """A function which takes the precision and recall of some model, and a value for beta,
    and returns the f_beta-score"""
    #.......................................................................
    return (1+beta**2) * precision * recall / (beta**2 * precision + recall)


# In[9]:


def f_score(precision, recall):
    return f_beta_score(precision, recall, beta=1)


# In[10]:


#initializing the model

if __name__ == '__main__':
    subject_filenames = find_csv_filenames(DATA_FOLDER)

    # new dataframe for performance metrics
    merged_subjects_df = pd.DataFrame()
    
    dict_t = {} #empty dictonary for Precision, Recall, F1Score (PRF)

    for name in subject_filenames:
        extracted_filename = os.path.basename(name)
        # split the filename by first underscore
        lhs, rhs = extracted_filename.split("_", 1)

        df = load_data(name)

        # pop out label from dataframe
        y = df.pop("class")

        # pre-processing data using Lower pass filter
        X = preprocessData(df.values)
        print(X.shape)

        # data splitting into training & testing dataset
        x_train, x_test, y_train, y_test = train_test_split(X, y.values, test_size=0.2, shuffle=True)

        # Reshape and normalize training data
        x_train = x_train.reshape(x_train.shape[0], 8, 20)
        x_train = x_train / 255.0

        x_test = x_test.reshape(x_test.shape[0], 8, 20)
        x_test = x_test / 255.0

        # Common Spatial Pattern (CSP)- feature extraction technique
        csp = CSP(n_components=3)

        print(x_train.shape, y_train.shape)

        x_train = csp.fit_transform(x_train, y_train)
        x_test = csp.transform(x_test)

        # Assemble a classifier
        lda = LinearDiscriminantAnalysis()

        # fit classifier
        lda.fit(x_train, y_train)

        train_accuracy = lda.score(x_train, y_train)
        test_accuracy = lda.score(x_test, y_test)

        print("TRAIN ACCURACY SCORE: ", train_accuracy)
        print("TEST ACCURACY SCORE: ", test_accuracy)

        test_pred= lda.predict(x_test)
        
        dict_t['participant '+str(lhs)] = {} #looping for PRF

        for t in range(2):  # Go over our 2 classes, left hand & right hand
            (TP, FP, FN) = class_scores(y_test, test_pred, t)
            p = TP / (TP + FP)
            r = TP / (TP + FN)
            f = f_score(p, r)
            
            dict_t['participant '+str(lhs)][str(t)]={}
            dict_t['participant '+str(lhs)][str(t)]['precision']={}
            dict_t['participant '+str(lhs)][str(t)]['recal']={}
            dict_t['participant '+str(lhs)][str(t)]['f_score']={}
            dict_t['participant '+str(lhs)][str(t)]['precision']=p
            dict_t['participant '+str(lhs)][str(t)]['recal']=r
            dict_t['participant '+str(lhs)][str(t)]['f_score']=f

            print('class: {:d}'.format(t))
            print('\tPrecision: {:04.3f}'.format(p))
            print('\tRecall: {:04.3f}'.format(r))
            print('\tF-score: {:04.3f}'.format(f))
            print('')

        # append result of subject id in dataframe
        subject_dic = {"subject_id": lhs, "train_acc": train_accuracy, "test_acc": test_accuracy}
        merged_subjects_df = merged_subjects_df.append(subject_dic, ignore_index=True)

    # write into csv file
    merged_subjects_df.to_csv(SUBJECT_WISE_PERFORMANCE_METRIC_CSV_Filter, index=True)
    
    print(dict_t)


# In[11]:


# this needs to be done to create a csv file to show Precision, Recall and f1 score (subject wise)


import pandas as pd

df_1 = pd.DataFrame(dict_t).T
df_1.to_csv ('PRF_CSP_LDA_Filter.csv', index = True)


# In[ ]:




