from dataset_utils import *
from algos import *

x_train, y_train, x_test, y_test = split_data(data_set_path = "Battery_RUL.csv", train_percent = 0.8)

predictions, r = knn_regression(3, x_train, y_train, x_test, y_test)
print(r)

predictions, r = svm_regression(x_train, y_train, x_test, y_test)
print(r)

predictions, r = linear_regression(x_train, y_train, x_test, y_test)
print(r)

predictions, r = decision_tree_regression(x_train, y_train, x_test, y_test)
print(r)