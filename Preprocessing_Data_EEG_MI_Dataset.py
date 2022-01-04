#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# this class is used to do pre_processing subject wise
# create new output folder in current folder. all the new csv files will be generated there


# In[1]:


# importing required libraries

import os
from os import listdir
import pandas as pd


# In[3]:


# making List of unwanted columns

UNWANTED_COLS = ["TimeStamp", "Trigger", "t", "t2"]


# In[2]:


# list all the subject wise csv files

def find_csv_filenames(path_to_dir, suffix=".csv"):
    filenames = listdir(path_to_dir)
    return [os.path.join(path_to_dir, filename) for filename in filenames if filename.endswith(suffix)]


# In[4]:


# preprocessing code block

if __name__ == '__main__':
    
    # find csv from the folder "dataset"
    subject_filenames = find_csv_filenames("dataset")

    # iterate through all subject files
    for name in subject_filenames:
        filename = os.path.basename(name)
        # print(filename)

        # split the filename by first underscore
        lhs, rhs = filename.split("_", 1)

        # word to search
        search = lhs
        # search every subject's csv files in a list
        result = [subject for subject in subject_filenames if search in subject]

        # merge each subject's csv file
        merged_df = pd.DataFrame()
        for subject_name in result:
            print(subject_name)

            # read csv
            df = pd.read_csv(subject_name)

            # filter on Timestamp, only having values in the range of 1 to 5
            df = df.query('1 <= TimeStamp <=5')

            # remove unwanted cols
            df.drop(UNWANTED_COLS, axis=1, inplace=True)

            # replace class 1, with 1 and  -1 with 0
            df['class'] = df['class'].map({1: 1, -1: 0})

            merged_df = merged_df.append(df, ignore_index=True)

        # remove unwanted cols
        merged_df.drop(["Unnamed: 0", "trial"], axis=1, inplace=True)

        # write into csv file
        merged_df.to_csv("output/{}_merged.csv".format(search), index=False)


# In[ ]:




