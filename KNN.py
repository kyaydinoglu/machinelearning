import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

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

# Determine the best value of k using cross-validation
k_values = range(1, 31)
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

# Find the optimal k
optimal_k = k_values[cv_scores.index(max(cv_scores))]
print(f"The optimal number of neighbors is {optimal_k}.")

# Plot cross-validation accuracy vs. k
plt.figure(figsize=(10, 6))
plt.plot(k_values, cv_scores, marker='o')
plt.xlabel('Number of Neighbors K')
plt.ylabel('Cross-Validated Accuracy')
plt.title('Cross-Validation Accuracy for K-Values')
plt.show()

# Train the KNN classifier with the optimal k
knn = KNeighborsClassifier(n_neighbors=optimal_k)
knn.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = knn.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
