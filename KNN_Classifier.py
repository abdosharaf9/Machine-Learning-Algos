from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from dataset_utils import discretize_dataset
from classification_utils import *


def knn_classification(k, x_train, y_train, x_test, y_test):
    knn = KNeighborsClassifier(k)
    knn.fit(x_train, y_train)
    predictions = knn.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    return accuracy, cm


def cross_val_knn_classification(k, x_train, y_train, numberOfIterations = 10):
    knn = KNeighborsClassifier(k)
    cv_accuracy_scores = cross_val_score(knn, x_train, y_train, cv = numberOfIterations, scoring=make_scorer(accuracy_score))
    return cv_accuracy_scores


x_train, y_train, x_test, y_test = discretize_dataset(dataset_path = "Battery_RUL.csv", train_percent = 0.8)
accuracy, cm = knn_classification(3, x_train, y_train, x_test, y_test)
sensitivity, specificity, precision, recall, true_positive_rate, false_positive_rate = calculate_multiclass_metrics(cm)


print("KNN Classifier:")
print(f"Accuracy = {accuracy:.4f}")
print(f"Confusion Matrix = \n{cm}")
print(f"Sensitivity = {sensitivity}")
print(f"Specificity = {specificity}")
print(f"Precision = {precision}")
print(f"Recall = {recall}")
print(f"True Positive Rate = {true_positive_rate}")
print(f"False Positive Rate = {false_positive_rate}")

print("\n===========================================\n")

cv_accuracy_scores = cross_val_knn_classification(3, x_train, y_train)
print("Cross validation on KNN Classifier:")
print(f"Mean Accuracy value using 10-Fold CV = {cv_accuracy_scores.mean():.4f}")