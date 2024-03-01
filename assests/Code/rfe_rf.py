import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from numpy import unique
# evaluate RFE for classification
from numpy import array
from pandas import read_csv
from numpy import mean
from numpy import std
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from imported_files.paths_n_vars import features_file , intra_annotated_file , all_features_file

# features = read_csv(features_file)
# labels = read_csv(intra_annotated_file).iloc[0] # this will exract the annotation 2nd row

rand_state = 54
test_fraction = 0.5
# re annotation
annotated_data = read_csv('5.Ten_sec_annotated_data/patient_0_1_10.csv')
labels = annotated_data.iloc[0] # this will exract the annotation 2nd row
features = (annotated_data.iloc[1 : , ])
x_train, x_test, y_train, y_test = train_test_split(features.T, labels, test_size = test_fraction, random_state = rand_state, stratify = labels)

# create pipeline
rfe = RFE(estimator = RandomForestClassifier(), n_features_to_select = 80)
model = RandomForestClassifier( n_estimators = 10 , criterion = 'entropy' , verbose = 1)
print("hello")
pipeline = Pipeline(steps=[('feature_selection',rfe),('model',model)] , verbose=True)
print("pipeline created")
# evaluate model
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1)
print("cv starts")
n_scores = cross_val_score(pipeline , x_train , y_train , scoring = 'accuracy' , cv = cv , n_jobs = -1 , error_score = 'raise' , verbose=2)
print("cv ends")
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
print(f"Confu mtrx = \n{confusion_matrix}")
print("\nClassification Report:\n")
print(metrics.classification_report(y_test, test_pred_rfe_rf))
print("\nAvg score on test dataset = {}".format(pipeline.score(x_test , y_test)))
print(pipeline)
print('Accuracy: %.5f (%.5f)' % (mean(n_scores), std(n_scores)))
pipeline.fit(features,labels)
selected_features_mask = pipeline.named_steps['feature_selection'].support_
selected_feature_names = [feature_name for feature_name, selected in zip(features.columns, selected_features_mask) if selected]
print(array(selected_feature_names))

