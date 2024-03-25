from imported_files.paths_n_vars import feature_merged

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from numpy import unique

# evaluate RFE for classification
from numpy import array
import pandas as pd
from numpy import mean
from numpy import std
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from collections import Counter

features = pd.read_csv(feature_merged)
labels = features['annotation'] # this will exract the annotation 2nd row

rand_state = 54
test_fraction = 0.5
if 'annotation' in features.columns :
    features.drop(['annotation'] , axis = 'columns' , inplace = True)
if 'PatientID' in features.columns :      
    features.drop(['PatientID'] , axis = 'columns', inplace = True)

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size = test_fraction, random_state = rand_state, stratify = labels)

# info abt dataset
print('_'*50)
print('Info Abt Dataset:')
print("Number of features: ", x_train.shape[1])
num_instances_train = dict(Counter(y_train))
num_instances_test = dict(Counter(y_test))
print(f"Total instances : {dict(Counter(labels))}")
print(f"Train instances : {num_instances_train}")
print(f"Test instances : {num_instances_test}")



# create pipeline
print('_'*50)
rfe = RFE(estimator = RandomForestClassifier(), n_features_to_select = 100)
model = RandomForestClassifier( n_estimators = 30 , max_depth= 10, criterion = 'gini' , verbose = 1, n_jobs=-1)
pipeline = Pipeline(steps=[('feature_selection',rfe),('model',model)] , verbose=True)
print("pipeline created")

# evaluate model
print('_'*50)
print("cv starts")
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1)
n_scores = cross_val_score(pipeline , x_train , y_train , scoring = 'accuracy' , cv = cv , n_jobs = -1 , error_score = 'raise' , verbose=1)
print('CV Accuracy: %.5f (%.5f)' % (mean(n_scores), std(n_scores)))
print("cv ends")

# fit to pipeline
print('_'*50)
print('Fitting data to pipeline')
pipeline.fit(x_train,y_train)

# report performance
test_pred_rfe_rf = pipeline.predict(x_test)
confusion_matrix = metrics.confusion_matrix(y_test , test_pred_rfe_rf)
class_list = unique(y_test)
ax = plt.axes()
sns.set_theme(font_scale=1.3)
sns.heatmap(confusion_matrix , annot = True , fmt = "g" , ax = ax , cmap = "magma")
ax.set_title('Confusion Matrix - Random Forest: ')
ax.set_xlabel("Predicted label" , fontsize = 15)
ax.set_xticklabels(class_list)
ax.set_ylabel("True Label", fontsize = 15)
ax.set_yticklabels(class_list, rotation = 0)
plt.show()

# classification report
print('_'*50)
print(f"Confu mtrx = \n{confusion_matrix}")
print("\nClassification Report:\n")
print(metrics.classification_report(y_test, test_pred_rfe_rf))
print("\nAvg score on test dataset = {}".format(pipeline.score(x_test , y_test)))
print(pipeline)

# print names of selected features
selected_features_mask = pipeline.named_steps['feature_selection'].support_
selected_feature_names = [feature_name for feature_name, selected in zip(x_train.columns, selected_features_mask) if selected]
print('_'*50)
print(f'{array(selected_feature_names) = }')
