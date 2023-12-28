import numpy as np

def calculate_multiclass_metrics(confusion_matrix):
    num_classes = len(confusion_matrix)
    
    sensitivity = np.zeros(num_classes)
    specificity = np.zeros(num_classes)
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    true_positive_rate = np.zeros(num_classes)
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
        true_positive_rate[i] = sensitivity[i]
        false_positive_rate[i] = FP / (TN + FP)

    return sensitivity, specificity, precision, recall, true_positive_rate, false_positive_rate