from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from numpy import unique

# Importing necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from imported_files.paths_n_vars import feature_merged
from pandas import read_csv

features = read_csv(feature_merged)
labels = labels = features['annotation'] # this will exract the annotation 2nd row

rand_state = 54
test_fraction = 0.5
if 'annotation' in features.columns :
        features.drop(['annotation'] , axis = 'columns' , inplace = True)
if 'PatientID' in features.columns :      
    features.drop(['PatientID'] , axis = 'columns', inplace = True)
    

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=test_fraction, random_state=rand_state, stratify=labels)

# Create pipeline
pipeline = Pipeline([
    ('feature_selection', RFE(estimator=RandomForestClassifier())),
    ('model', RandomForestClassifier())
])

# Define parameter grid for RandomizedSearchCV
param_grid = {
    'feature_selection__n_features_to_select': randint(30, 100),  # Randomly select n_features_to_select from 1 to 100
    'model__n_estimators': randint(15, 100),  # Randomly select n_estimators from 10 to 100
    'model__max_depth': randint(5, 20),  # Randomly select max_depth from 5 to 20
    'model__criterion': ['gini', 'entropy']  # Select either 'gini' or 'entropy'
}

# Create RandomizedSearchCV object

cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=1, random_state=1)
random_search = RandomizedSearchCV(estimator=pipeline, param_distributions=param_grid, n_iter=1,
                                   scoring='accuracy', cv=cv, verbose=2, random_state=42, n_jobs=-1)

# Perform the random search
random_search.fit(x_train, y_train)

# Get the best parameters
best_params = random_search.best_params_
print("Best Parameters:", best_params)

# Get the best estimator
best_estimator = random_search.best_estimator_

# Evaluate the best estimator
test_pred_rfe_rf = best_estimator.predict(x_test)
confusion_matrix = metrics.confusion_matrix(y_test, test_pred_rfe_rf)

# Plot confusion matrix
class_list = unique(y_test)
ax = plt.axes()
sns.set_theme(font_scale=1.3)
sns.heatmap(confusion_matrix, annot=True, fmt="g", ax=ax, cmap="magma")
ax.set_title('Confusion Matrix - Random Forest: ')
ax.set_xlabel("Predicted label", fontsize=15)
ax.set_xticklabels(class_list)
ax.set_ylabel("True Label", fontsize=15)
ax.set_yticklabels(class_list, rotation=0)
plt.show()

# Print classification report
print(f"Confusion Matrix:\n{confusion_matrix}")
print("\nClassification Report:\n")
print(metrics.classification_report(y_test, test_pred_rfe_rf))
print("\nAverage score on test dataset =", best_estimator.score(x_test, y_test))

# Print names of selected features
selected_features_mask = best_estimator.named_steps['feature_selection'].support_
