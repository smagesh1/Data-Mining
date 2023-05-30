import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
# Part 1: Decision Trees with Categorical Attributes

# Return a pandas dataframe containing the data set that needs to be extracted from the data_file.
# data_file will be populated with the string 'adult.csv'.
def read_csv_1(data_file):
    df = pd.read_csv("adult.csv")
    df = df.drop('fnlwgt', axis=1)
    return df


# Return the number of rows in the pandas dataframe df.
def num_rows(df):
    df = df.shape[0]
    return df

# Return a list with the column names in the pandas dataframe df.
def column_names(df):
    df = list(df.columns)
    return df

# Return the number of missing values in the pandas dataframe df.
def missing_values(df):
    df = df.isnull().values.sum()
    return df

# Return a list with the columns names containing at least one missing value in the pandas dataframe df.
def columns_with_missing_values(df):
    return list(df.columns[df.isna().any()])

# Return the percentage of instances corresponding to persons whose education level is 
# Bachelors or Masters, by rounding to the third decimal digit,
# in the pandas dataframe df containing the data set in the adult.csv file.
# For example, if the percentage is 0.21547%, then the function should return 0.216.
def bachelors_masters_percentage(df):
    df = len(df[(df['education'] == 'Bachelors') | (df['education'] == 'Masters')])/len(df)
    return round(df,3)*100


# Return a pandas dataframe (new copy) obtained from the pandas dataframe df 
# by removing all instances with at least one missing value.
def data_frame_without_missing_values(df):
    df = df.dropna()
    return df


# Return a pandas dataframe (new copy) from the pandas dataframe df 
# by converting the df categorical attributes to numeric using one-hot encoding.
# The function should not encode the target attribute, and the function's output
# should not contain the target attribute.
def one_hot_encoding(df):
    return pd.get_dummies(df.loc[:,df.columns != 'class'])
    

# Return a pandas series (new copy), from the pandas dataframe df, 
# containing only one column with the labels of the df instances
# converted to numeric using label encoding. 
def label_encoding(df):
    labelencoder = preprocessing.LabelEncoder()
    series_encodedvalues = labelencoder.fit_transform(df['class'])
    return pd.Series(series_encodedvalues)


# Given a training set X_train containing the input attribute values 
# and labels y_train for the training instances,
# build a decision tree and use it to predict labels for X_train. 
# Return a pandas series with the predicted values. 
def dt_predict(X_train,y_train):
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    return pd.Series(clf.predict(X_train))
    



# Given a pandas series y_pred with the predicted labels and a pandas series y_true with the true labels,
# compute the error rate of the classifier that produced y_pred.  
def dt_error_rate(y_pred, y_true):
    error_rate = 1 - accuracy_score(y_true, y_pred)
    return error_rate




