from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import accuracy_score, r2_score,make_scorer,confusion_matrix
from sklearn.model_selection import cross_val_score
from dataset_utils import *
from algos import *


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


def knn_regressor_with_cv(k, x_train, y_train, numberOfIterations = 10):
    knn = KNeighborsRegressor(n_neighbors = k)
    r2_scorer = make_scorer(r2_score)
    cv_scores = cross_val_score(knn, x_train, y_train, cv = numberOfIterations, scoring=r2_scorer)
    mean_r2 = cv_scores.mean()
    return mean_r2


def knn_classification_with_cv(k, x_train, y_train, numberOfIterations = 10):
    knn = KNeighborsClassifier(n_neighbors = k)
    r2_scorer = make_scorer(r2_score)
    cv_scores = cross_val_score(knn, x_train, y_train, cv = numberOfIterations, scoring=r2_scorer)
    mean_r2 = cv_scores.mean()
    return mean_r2


def knn_classification_with_confusion_matrix(k, x_train, y_train, x_test, y_test):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    predictions = knn.predict(x_test)
    cm = confusion_matrix(y_test, predictions)
    return cm


def calculate_multiclass_metrics(confusion_matrix):
    num_classes = len(confusion_matrix)
    
    sensitivity = np.zeros(num_classes)
    specificity = np.zeros(num_classes)
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    positive_rate = np.zeros(num_classes)
    false_positive_rate = np.zeros(num_classes)

    for i in range(num_classes):
        TP = confusion_matrix[i, i]
        FP = np.sum(confusion_matrix[:, i]) - TP
        FN = np.sum(confusion_matrix[i, :]) - TP
        TN = np.sum(confusion_matrix) - TP - FN - FP

        sensitivity[i] = TP / (TP + FN)
        specificity[i] = TN / (TN + FP)
        precision[i] = TP / (TP + FP)
        recall[i] = sensitivity[i]
        positive_rate[i] = np.sum(confusion_matrix[:, i]) / np.sum(confusion_matrix)
        false_positive_rate[i] = FP / (TN + FP)

    return sensitivity, specificity, precision, recall, positive_rate, false_positive_rate


x_train_original, y_train_original, x_test_original, y_test_original = split_data(data_set_path = "Battery_RUL.csv", train_percent = 0.8)

_,knn_regression_r2 = knn_regression(k=3, x_train = x_train_original, y_train = y_train_original, x_test = x_test_original, y_test = y_test_original)
print("K-NN Regression Algorithm R^2 = ",knn_regression_r2)
print("=====================================================================")

knn_regression_r2_with_cv = knn_classification_with_cv(k = 3, x_train = x_train_original, y_train = y_train_original)
print("K-NN Regression Algorithm After 10-Flod CV = ",knn_regression_r2_with_cv)
print("=====================================================================")

x_train_discretized, y_train_discretized, x_test_discretized, y_test_discretized = discretize_dataset(data_set_path = "Battery_RUL.csv", train_percent = 0.8)

_,knn_classification_r2=knn_classification(k=3,x_train=x_train_discretized,y_train=y_train_discretized,x_test=x_test_discretized,y_test=y_test_discretized)
print("K-NN Classification Algorithm R^2 = ",knn_classification_r2)
print("=====================================================================")

knn_classification_r2_with_cv = knn_classification_with_cv(k = 3, x_train = x_train_discretized, y_train = y_train_discretized)
print("K-NN Classification Algorithm After 10-Flod CV = ",knn_classification_r2_with_cv)
print("=====================================================================")

confusion_matrix = knn_classification_with_confusion_matrix(k=3,x_train=x_train_discretized,y_train=y_train_discretized,x_test=x_test_discretized,y_test=y_test_discretized)
print("Confusion Matrix  \n\n",confusion_matrix,"\n")
print("=====================================================================")


sensitivity, specificity, precision, recall, positive_rate, false_positive_rate = calculate_multiclass_metrics(confusion_matrix)
print("Senstivity = ",sensitivity)
print("=====================================================================")
print("Specificity = ",specificity)
print("=====================================================================")
print("Precision = ",precision)
print("=====================================================================")
print("Recall = ",recall)
print("=====================================================================")
print("True Positive Rate = ",positive_rate)
print("=====================================================================")
print("False Positive Rate = ",false_positive_rate)
print("=====================================================================")