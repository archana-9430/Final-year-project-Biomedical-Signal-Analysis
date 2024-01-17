feature_csv_path = "features_125Hz\\bidmc03m.csv"
annotated_csv_path = "Annotated_125Hz\\bidmc03m.csv"

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Load features file with segment numbers and feature values
features = pd.read_csv(feature_csv_path, header=None)  # Load without header

# Load annotations file
annotations = pd.read_csv(annotated_csv_path, header=None).iloc[0]

features = features.iloc[1:, 1:]  # Remove first row and first column

# Transpose the features matrix for compatibility with annotations
# features = features.T

# Assign column names to features matrix
features.columns = range(1, len(features.columns) + 1)  # Assign numerical column names

# Extract features (X) and annotations (y)
X = features.values  # Features matrix
y = annotations.values  # Target annotations

print("Shape of X (features matrix):", X.shape)
print("Shape of y (target annotations):", y.shape)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=1000, random_state=42)

# Train the classifier
rf_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_classifier.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of Random Forest Classifier: {accuracy}")

importances = rf_classifier.feature_importances_
feature_names = features.columns

feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

print(feature_importances)

