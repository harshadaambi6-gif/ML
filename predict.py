from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve
)

# True labels and predicted labels
y_true = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0]
y_pred = [0, 1, 0, 0, 1, 1, 0, 1, 1, 1]

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)

# Accuracy
accuracy = accuracy_score(y_true, y_pred)
print("Accuracy:", accuracy)

# Precision
precision = precision_score(y_true, y_pred)
print("Precision:", precision)

# Recall
recall = recall_score(y_true, y_pred)
print("Recall:", recall)

# F1 Score
f1 = f1_score(y_true, y_pred)
print("F1 Score:", f1)

# Precision-Recall curve
precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred)
print("Precision-Recall Curve:")
print("Precisions:", precisions)
print("Recalls:", recalls)
print("Thresholds:", thresholds)