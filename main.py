import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load and process the dataset
file_path = 'diabetes_dataset.csv'
data = pd.read_csv(file_path, delimiter='\t', quotechar='"', header=None)

# Split the string into separate columns
data = data[0].str.split('\t', expand=True)

# Assign column names
data.columns = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 
    'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
]

# Convert columns to numeric
data = data.apply(pd.to_numeric)

# Display the first few rows and summary statistics
print(data.head())
print(data.describe())

# Split the dataset into training and testing sets
X = data.drop('Outcome', axis=1)
y = data['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train and evaluate MLPClassifier
mlp = MLPClassifier(max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)

# Compute metrics for MLP
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
conf_matrix_mlp = confusion_matrix(y_test, y_pred_mlp)
class_report_mlp = classification_report(y_test, y_pred_mlp)
roc_auc_mlp = roc_auc_score(y_test, mlp.predict_proba(X_test)[:, 1])

# Print MLP results
print("MLP Classifier Results")
print("Accuracy:", accuracy_mlp)
print("Classification Report:\n", class_report_mlp)
print("Confusion Matrix:\n", conf_matrix_mlp)
print("ROC AUC Score:", roc_auc_mlp)

# Train and evaluate SVMClassifier
svm = SVC(kernel='linear', probability=True, random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

# Compute metrics for SVM
accuracy_svm = accuracy_score(y_test, y_pred_svm)
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
class_report_svm = classification_report(y_test, y_pred_svm)
roc_auc_svm = roc_auc_score(y_test, svm.predict_proba(X_test)[:, 1])

# Print SVM results
print("SVM Classifier Results")
print("Accuracy:", accuracy_svm)
print("Classification Report:\n", class_report_svm)
print("Confusion Matrix:\n", conf_matrix_svm)
print("ROC AUC Score:", roc_auc_svm)

# Plot ROC curves for both classifiers
fpr_mlp, tpr_mlp, _ = roc_curve(y_test, mlp.predict_proba(X_test)[:, 1])
fpr_svm, tpr_svm, _ = roc_curve(y_test, svm.predict_proba(X_test)[:, 1])

plt.figure(figsize=(10, 6))
plt.plot(fpr_mlp, tpr_mlp, label=f'MLP (AUC = {roc_auc_mlp:.2f})')
plt.plot(fpr_svm, tpr_svm, label=f'SVM (AUC = {roc_auc_svm:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

# Plot confusion matrices for both classifiers
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

sns.heatmap(conf_matrix_mlp, annot=True, fmt='d', ax=axes[0], cmap='Blues')
axes[0].set_title('MLP Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

sns.heatmap(conf_matrix_svm, annot=True, fmt='d', ax=axes[1], cmap='Blues')
axes[1].set_title('SVM Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.show()
