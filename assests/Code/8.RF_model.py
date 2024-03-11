from imported_files.paths_n_vars import features_file, intra_annotated_file, all_features_file, ae_features_file
# import features_scaler as fScaler

rand_state = 54
test_fraction = 0.5
num_trees = 15
split_criteria = "gini"

# k of k fold cross validation
k = 9 # change if you want to experiment

# class list
# class_list = ["0" , "1"] # good segn = 0 , corrupted signal = 1

# ~~~~~~~LIBRARIES~~~~~~~
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# for cross validation
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score

# for creating confusion matrix
from sklearn import metrics

# to plot confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns # seaborn for additional functionality such as heat map

# additionals
from sklearn.metrics import classification_report

# to calculate number of instances of 0 and 1
from collections import Counter
# ~~~~~~~~~~~~END LIBRARIES~~~~~~~~~~~~~~~~~~~~~

def create_train_classifier(x_train_data , y_train_data):
    return RandomForestClassifier(n_estimators = num_trees , random_state = rand_state \
                                  , criterion = split_criteria, verbose = 1).fit(x_train_data , y_train_data)
    
def k_fold_s_crossval(x_train_data , y_train_data , k_value , classifier):
    rskf = RepeatedStratifiedKFold(n_splits = k_value , n_repeats = k_value , random_state = rand_state)
    result = cross_val_score(classifier , x_train_data , y_train_data , cv = rskf , verbose = 1 , n_jobs = -1)
    print("Cross validation Accuracy : mean = {} :: std = {}".format(result.mean() , result.std()))


def test_n_results(x_test_data , y_test_data , classifier , description:str = ""):

    # Now TESTING the model and showing the results:
    # confusion matrix 
    test_pred_decision_tree = classifier.predict(x_test_data)
    confusion_matrix = metrics.confusion_matrix(y_test_data , test_pred_decision_tree)
    class_list = np.unique(y_test_data)
    ax = plt.axes()
    sns.set_theme(font_scale=1.3)
    sns.heatmap(confusion_matrix , annot = True , fmt = "g" , ax = ax , cmap = "magma")
    ax.set_title('Confusion Matrix - Random Forest: ' + description)
    ax.set_xlabel("Predicted label" , fontsize = 15)
    ax.set_xticklabels(class_list)
    ax.set_ylabel("True Label", fontsize = 15)
    ax.set_yticklabels(class_list, rotation = 0)
    plt.show()

    # # roc curve plot
    # ax = plt.gca()
    # rfc_disp = metrics.RocCurveDisplay.from_estimator(classifier, x_test_data, y_test_data, ax=ax, alpha=0.8)
    # plt.show()

    # classification report
    print(f"Confu mtrx = \n{confusion_matrix}")
    print("\nClassification Report:\n")
    print(classification_report(y_test_data, test_pred_decision_tree))
    print("\nAvg score on test dataset = {}".format(classifier.score(x_test_data , y_test_data)))
    print(classifier)
    return test_pred_decision_tree

def rf_model( local_features_file, annotated_file , description : str = ""):
    # get the dataset from the files
    features = pd.read_csv(local_features_file)
    # labels = pd.read_csv(annotated_file).iloc[0] # this will exract the annotation 2nd row
    labels = features['annotation']  # this will exract the annotation col

    if local_features_file == annotated_file:
        features.drop(index=0, inplace=True)
        features = features.T
        
    if 'annotation' in features.columns :
        features.drop(['annotation'] , axis = 'columns' , inplace = True)

    print("Features df:\n" , features )
    # assert not np.any(np.isnan(features.values)) , "ERROR::FEATURES DATAFRAME HAS NAN VALUES"
    # split the dataset using test_train_split() function
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size = test_fraction, random_state = rand_state, stratify = labels)
    # features_train, features_test, y_train, y_test = train_test_split(features, labels, test_size = test_fraction, random_state = rand_state, stratify = labels)

    # pID_train = features_train['PatientID']
    # pID_test = features_test['PatientID']
    
    # x_train = features_train.drop(['PatientID'] , axis = 'columns')
    # x_test = features_test.drop(['PatientID'] , axis = 'columns')
    print("Number of features: ", features.shape[1] - 1)
    num_instances_train = dict(Counter(y_train))
    num_instances_test = dict(Counter(y_test))
    print(f"Train instances : {num_instances_train}")
    print(f"Test instances : {num_instances_test}")

    clf = create_train_classifier(x_train , y_train)
    k_fold_s_crossval(x_train , y_train , k , clf)
    y_pred = test_n_results(x_test , y_test , clf , description)
    actual_pred_tuple = tuple(zip(y_test.values , y_pred))
    # classification_rep = {pID_test.values : actual_pred_tuple}
    # print(classification_rep)

# print("\n~~~~~ RF:: directly feeding sample values ~~~~~")
# rf_model(intra_annotated_file , intra_annotated_file , description = "Sample values")
# print("\n~~~~~ RF:: W/O AE FEATURES ~~~~~")p
# rf_model(features_file , intra_annotated_file , description = "Statistical features")
print("\n~~~~~ RF:: WITH ALL FEATURES ~~~~~")
# rf_model( all_features_file , intra_annotated_file , description = "All features")
rf_model( '6.Features_extracted/all_features_re_anno.csv' , intra_annotated_file , description = "All features")
# print("\n~~~~~ RF:: WITH AE FEATURES ONLY ~~~~~")
# rf_model( ae_features_file, intra_annotated_file , description = "with AE features only")

#~~~~~~~~~ AFTER REANNOTATION ~~~~~~~~~~
# print("\n~~~~~ RF:: directly feeding sample values :: RE ANNOTATION ~~~~~")
# rf_model('5.New_annotated_data/merged.csv' , '5.New_annotated_data/merged.csv' , description = "Sample values")
# print("\n~~~~~ RF:: W/O AE FEATURES :: RE ANNOTATION ~~~~~")
# rf_model('6.Features_extracted/features_filtered_1_1_10.csv' , '5.New_annotated_data/merged.csv' , description = "Statistical features")
# print("\n~~~~~ RF::All FEATURES : ANNOTATION ~~~~~")
# rf_model('6.Features_extracted/all_features_re_anno_2d.csv' , '' , description = "All features")
# print("\n~~~~~ RF:: WITH AE FEATURES ONLY ~~~~~")
# rf_model( ae_features_file, '5.New_annotated_data/merged.csv' , description = "with AE features only")