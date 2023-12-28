from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, r2_score, mean_squared_error, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from dataset_utils import split_data


def knn_regression(k, x_train, y_train, x_test, y_test):
    knn = KNeighborsRegressor(k)
    knn.fit(x_train, y_train)
    predictions = knn.predict(x_test)
    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    return r2, mse, mae


def cross_val_knn_regression(k, x_train, y_train, numberOfIterations = 10):
    knn = KNeighborsRegressor(k)
    cv_r2_scores = cross_val_score(knn, x_train, y_train, cv = numberOfIterations, scoring=make_scorer(r2_score))
    cv_mse_scores = cross_val_score(knn, x_train, y_train, cv = numberOfIterations, scoring=make_scorer(mean_squared_error))
    cv_mae_scores = cross_val_score(knn, x_train, y_train, cv = numberOfIterations, scoring=make_scorer(mean_absolute_error))
    return cv_r2_scores, cv_mse_scores, cv_mae_scores


x_train, y_train, x_test, y_test = split_data(dataset_path = "Battery_RUL.csv", train_percent = 0.8)
r2, mse, mae = knn_regression(3, x_train, y_train, x_test, y_test)


print("KNN Regressor:")
print(f"R^2 = {r2:.4f}")
print(f"Mean Squared Error = {mse:.4f}")
print(f"Mean Absolute Error = {mae:.4f}")

print("\n===========================================\n")

cv_r2_scores, cv_mse_scores, cv_mae_scores = cross_val_knn_regression(3, x_train, y_train)
print("Cross validation on KNN Regressor:")
print(f"Mean R^2 value using 10-Fold CV = {cv_r2_scores.mean():.4f}")
print(f"Mean MSE value using 10-Fold CV = {cv_mse_scores.mean():.4f}")
print(f"Mean MAE value using 10-Fold CV = {cv_mae_scores.mean():.4f}")