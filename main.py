from dataset_utils import *
from algos import *

x_train, y_train, x_test, y_test = split_data(data_set_path = "Battery_RUL.csv", train_percent = 0.8)

predictions, r2 = knn_regression(3, x_train, y_train, x_test, y_test)
print(r2)

predictions, r2 = svm_regression(x_train, y_train, x_test, y_test)
print(r2)

predictions, r2 = linear_regression(x_train, y_train, x_test, y_test)
print(r2)

predictions, r2 = decision_tree_regression(x_train, y_train, x_test, y_test)
print(r2)