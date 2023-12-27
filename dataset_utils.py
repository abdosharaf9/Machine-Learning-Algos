import pandas as pd

# 15064 rows
# data = pd.read_csv("Battery_RUL.csv")
# rows = data.shape[0]
# print(round(0.8 * rows))


def split_data(data_set_path, train_percent):
    data = pd.read_csv(data_set_path)
    rows = data.shape[0]
    train_end_index = round(train_percent * rows)
    x_train = data.iloc[:train_end_index, :-1].values
    y_train = data.iloc[:train_end_index, -1].values
    
    x_test = data.iloc[train_end_index:, :-1].values
    y_test = data.iloc[train_end_index:, -1].values
    
    return x_train, y_train, x_test, y_test