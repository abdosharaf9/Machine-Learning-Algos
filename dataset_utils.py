import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

# 15064 rows
# data = pd.read_csv("Battery_RUL.csv")
# rows = data.shape[0]
# print(round(0.8 * rows))


def split_data(dataset_path, train_percent):
    data = pd.read_csv(dataset_path)
    rows = data.shape[0]
    train_end_index = round(train_percent * rows)
    x_train = data.iloc[:train_end_index, :-1].values
    y_train = data.iloc[:train_end_index, -1].values
    
    x_test = data.iloc[train_end_index:, :-1].values
    y_test = data.iloc[train_end_index:, -1].values
    
    return x_train, y_train, x_test, y_test


def discretize_dataset(dataset_path, train_percent, n_bins=5, encode='ordinal', strategy='uniform'):
    datafram = pd.read_csv(dataset_path)
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy, subsample=None)
    data = pd.DataFrame(discretizer.fit_transform(datafram), columns=datafram.columns)
    
    rows = data.shape[0]
    train_end_index = round(train_percent * rows)
    x_train = data.iloc[:train_end_index, :-1].values
    y_train = data.iloc[:train_end_index, -1].values
    
    x_test = data.iloc[train_end_index:, :-1].values
    y_test = data.iloc[train_end_index:, -1].values
    
    return x_train, y_train, x_test, y_test