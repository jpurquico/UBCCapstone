#!/usr/bin/env python
# coding: utf-8

# # Preprocessing

# - This file collects preprocessing from travel_bert.ipynb, LinearSVC.ipynb and a tutorial from COLX_585. 
# - The input is data path and domain name. The output is vectorized batches.


import pandas as pd
from sklearn.model_selection import train_test_split

# ## 1. Remove unwanted columns then convert back to CSV

def preprocess_csv(input_file_name, output_file_name, file_directory="../../data/"):
    """
    cleans a csv file by converting it into a Pandas dataframe and removing unwanted columns, 
    then converts it to another csv file for BERT multi-label classification.
    """
    df = pd.read_csv(file_directory+input_file_name)
    df = df[['Comment', 'Tags']].to_csv(file_directory+output_file_name, index=False)
    return f"{file_directory+output_file_name}"  

# ## 2. Read data

def read_data(path, checked = False):
    """
    This function reads a csv file and filters out na and non-freq tags.
    
    Parameters: 
    ------------
        path: data path
        checked: boolean (True-only use checked data, False-use all data)
    Return:
    ------------
        df: filtered data
    """
    df = pd.read_csv(path)
    df = df.dropna() # drop na
    if checked: # only use the checked data 
        df = df[df['Tags confirmed']=='checked'] 
    # remove tags that occur only once
    value_counts = df['Tags'].value_counts()
    remove_rows = value_counts[value_counts < 2].index
    df = df[~df.Tags.isin(remove_rows)]
    return df

# ## 3. Splitting

def split_and_write(df, domain, test_size=0.2):
    """
    This function splits train and test dataframe and write them to csv files.
    
    Parameters: 
    ------------
        df: data
        test_size: proportion of test 
        domain: domain name (vaccine, travel, etc)
    """
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    train_df.to_csv(f"../data/{domain}_train.csv",columns=["Comment", "Tags"], index=False)
    test_df.to_csv(f"../data/{domain}_test.csv",columns=["Comment", "Tags"], index=False)