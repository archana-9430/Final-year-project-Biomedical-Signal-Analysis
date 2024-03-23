import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load your features CSV file
data = pd.read_csv('stats_features.csv')
data = data.iloc[:, 1:]

# Extract annotations from the second column
annotations = data.iloc[:, 0]
features = data.iloc[:, 1:]
# print("annotation", annotations)
# print("features", features)

# Encode annotations if they are not numeric
# if annotations.dtype != np.number:
#     label_encoder = LabelEncoder()
#     annotations = label_encoder.fit_transform(annotations)

# Split the dataset into features and target
X = features.values
y = annotations

# Get the feature names from the header
feature_names = features.columns

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Load your pre-trained Random Forest model
model = RandomForestClassifier(n_estimators=15, criterion='gini', random_state=42)  # Define your model with 15 trees and Gini impurity

# Train the model
model.fit(X_train, y_train)


def Explainer():
    # Define a function to be passed to the explainer
    def model_predict(X):
        return model.predict_proba(X)

    # Use SHAP to compute SHAP values for the pre-trained model
    explainer = shap.Explainer(model_predict, X_train)
    shap_values = explainer.shap_values(X_test)

    # Calculate mean absolute SHAP values for each feature
    mean_abs_shap_values = np.abs(shap_values).mean(axis=0)
    print('mean_abs_shap_values', mean_abs_shap_values)


    # Identify features with low importance
    threshold = 0.0037 # Define a threshold for low importance
    redundant_features = [feature_names[i] for i, importance in enumerate(mean_abs_shap_values) if np.any(importance < threshold)]

    mask = ~np.isin(feature_names, redundant_features)

    # Remove redundant features from the datasets
    X_train_min = X_train[:, mask]
    X_test_min = X_test[:, mask]

    # # Remove redundant features from the dataset
    # X_train_min = pd.DataFrame(X_train).drop(columns=[col for col in redundant_features if col in feature_names], errors='ignore')
    # X_test_min = pd.DataFrame(X_test).drop(columns=[col for col in redundant_features if col in feature_names], errors='ignore')

    # Retrain the model without redundant features
    model_min = RandomForestClassifier(n_estimators=15, criterion='gini', random_state=42)  # Define the model with the same parameters
    model_min.fit(X_train_min, y_train)

    # Evaluate model performance
    score_before = model.score(X_test, y_test)
    score_after = model_min.score(X_test_min, y_test)

    print("Model performance of Explainer:")
    print(f"Before feature minimization: Accuracy = {score_before:.4f}")
    print(f"After feature minimization: Accuracy = {score_after:.4f}")

    print("Redundant features:", redundant_features)
    
    
    
def TreeExplainer():
    # Use TreeExplainer to compute SHAP values for the pre-trained model
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Calculate mean absolute SHAP values for each feature
    mean_abs_shap_values = np.abs(shap_values).mean(axis=0)
    print('mean_abs_shap_values', mean_abs_shap_values)


    # Identify features with low importance
    threshold = 0.0041 # Define a threshold for low importance
    redundant_features = [feature_names[i] for i, importance in enumerate(mean_abs_shap_values) if np.any(importance < threshold)]

    mask = ~np.isin(feature_names, redundant_features)

    # Remove redundant features from the datasets
    X_train_min = X_train[:, mask]
    X_test_min = X_test[:, mask]

    # # Remove redundant features from the dataset
    # X_train_min = pd.DataFrame(X_train).drop(columns=[col for col in redundant_features if col in feature_names], errors='ignore')
    # X_test_min = pd.DataFrame(X_test).drop(columns=[col for col in redundant_features if col in feature_names], errors='ignore')

    # Retrain the model without redundant features
    model_min = RandomForestClassifier(n_estimators=15, criterion='gini', random_state=42)  # Define the model with the same parameters
    model_min.fit(X_train_min, y_train)

    # Evaluate model performance
    score_before = model.score(X_test, y_test)
    score_after = model_min.score(X_test_min, y_test)

    print("Model performance of TreeExplainer:")
    print(f"Before feature minimization: Accuracy = {score_before:.4f}")
    print(f"After feature minimization: Accuracy = {score_after:.4f}")

    print("Redundant features:", redundant_features)
    
# Explainer()
TreeExplainer()