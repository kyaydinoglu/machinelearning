import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

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

print("MLP Classifier Results")
print("Accuracy:", accuracy_score(y_test, y_pred_mlp))
print("Classification Report:\n", classification_report(y_test, y_pred_mlp))

# Train and evaluate SVMClassifier
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

print("SVM Classifier Results")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Classification Report:\n", classification_report(y_test, y_pred_svm))
