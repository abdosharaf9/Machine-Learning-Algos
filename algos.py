from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
from math import sqrt


def knn_regression(k, x_train, y_train, x_test, y_test):
    knn = KNeighborsRegressor(n_neighbors = k)
    knn.fit(x_train, y_train)
    predictions = knn.predict(x_test)
    r2 = r2_score(y_test, predictions)
    return predictions, r2


def knn_classification(k, x_train, y_train, x_test, y_test):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(x_train, y_train)
    predictions = knn.predict(x_test)
    score = accuracy_score(y_test, predictions)
    return predictions, score


def svm_regression(x_train, y_train, x_test, y_test):
    svr = SVR()
    svr.fit(x_train, y_train)
    predictions = svr.predict(x_test)
    r2 = r2_score(y_test, predictions)
    return predictions, r2


def svm_classification(x_train, y_train, x_test, y_test):
    svc = SVC()
    svc.fit(x_train, y_train)
    predictions = svc.predict(x_test)
    score = accuracy_score(y_test, predictions)
    return predictions, score


def linear_regression(x_train, y_train, x_test, y_test):
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    predictions = lr.predict(x_test)
    r2 = r2_score(y_test, predictions)
    return predictions, r2


def decision_tree_regression(x_train, y_train, x_test, y_test):
    tree = DecisionTreeRegressor()
    tree.fit(x_train, y_train)
    predictions = tree.predict(x_test)
    r2 = r2_score(y_test, predictions)
    return predictions, r2


def decision_tree_classification(x_train, y_train, x_test, y_test):
    tree = DecisionTreeClassifier()
    tree.fit(x_train, y_train)
    predictions = tree.predict(x_test)
    score = accuracy_score(y_test, predictions)
    return predictions, score