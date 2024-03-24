from imported_files.paths_n_vars import features_file, intra_annotated_file, all_features_file, ae_features_file

rand_state = 54
test_fraction = 0.5

# k of k flod cross validation
k = 5 # change if you want to experiment

# class list
# class_list = ["0" , "1"] # good segn = 0 , corrupted signal = 1

# ~~~~~~~LIBRARIES~~~~~~~
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
#import naive bayes classifier
from sklearn.naive_bayes import GaussianNB

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
    # fit the decision tree model into the train set
    classifier = GaussianNB()
    classifier.fit(x_train_data , y_train_data)
    return classifier

def k_fold_s_crossval(x_train_data , y_train_data , k_value , classifier):
    # k fold cross validation
    rskf = RepeatedStratifiedKFold(n_splits = k_value , n_repeats = k_value , random_state = rand_state)
    result = cross_val_score(classifier , x_train_data , y_train_data , cv = rskf)
    print("Cross validation Accuracy : mean = {} :: std = {}".format(result.mean() , result.std()))


def test_n_results(x_test_data , y_test_data , classifier , description:str = ""):

    # Now TESTING the model and showing the results:
    # confusion matrix 
    test_pred_Gaussian_NB = classifier.predict(x_test_data)
    confusion_matrix = metrics.confusion_matrix(y_test_data , test_pred_Gaussian_NB)
    class_list = np.unique(y_test_data)
    ax = plt.axes()
    sns.set_theme(font_scale=1.3)
    sns.heatmap(confusion_matrix , annot = True , fmt = "g" , ax = ax , cmap = "magma")
    ax.set_title('Confusion Matrix - Gaussian Naive Bayes: ' + description)
    ax.set_xlabel("Predicted label" , fontsize = 15)
    ax.set_xticklabels(class_list)
    ax.set_ylabel("True Label", fontsize = 15)
    ax.set_yticklabels(class_list, rotation = 0)
    plt.show()

    # classification report
    print("\nClassification Report:\n")
    print(classification_report(y_test_data, test_pred_Gaussian_NB))
    print("\nAvg score on test dataset = {}".format(classifier.score(x_test_data , y_test_data)))
    print(classifier)

def GaussianNB_model( local_features_file, annotated_file : str = ""  , description : str = ""):
     # get the dataset from the files
    features = pd.read_csv(local_features_file)
    labels = pd.read_csv(annotated_file).iloc[0] # this will exract the annotation 2nd row

    if local_features_file == all_features_file and 'annotation' in features.columns :
        features.drop(['annotation'] , axis = 'columns' , inplace = True)
    
    assert not np.any(np.isnan(features)) , "ERROR::FEATURES DATAFRAME HAS NAN VALUES"
    # split the dataset using test_train_split() function
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size = test_fraction, random_state = rand_state, stratify = labels)

    num_instances_train = dict(Counter(y_train))
    num_instances_test = dict(Counter(y_test))
    print(f"Train instances : {num_instances_train}")
    print(f"Test instances : {num_instances_test}")

    clf = create_train_classifier(x_train , y_train)
    k_fold_s_crossval(x_train , y_train , k , clf)
    test_n_results(x_test , y_test , clf , description)

# print("\n~~~~~ Gaussian NB:: W/O AE FEATURES ~~~~~")
# rf_model(features_file , intra_annotated_file , description = "w/o AE features")
# print("\n~~~~~ Gaussian NB:: WITH ALL FEATURES ~~~~~")
# rf_model( all_features_file , description = "with all features")

#~~~~~~~~~ AFTER REANNOTATION ~~~~~~~~~~
print("\n~~~~~ GaussianNB:: W/O AE FEATURES :: RE ANNOTATION ~~~~~")
GaussianNB_model('6.Features_extracted/features_filtered_1_1_10.csv' , '5.Ten_sec_annotated_data/patient_0_1_10.csv' , description = "Statistical features")
print("\n~~~~~ GaussianNB::All FEATURES :: RE ANNOTATION ~~~~~")
GaussianNB_model('6.Features_extracted/all_features.csv' , '5.Ten_sec_annotated_data/patient_0_1_10.csv' , description = "All features")
print("\n~~~~~ GaussianNB:: WITH AE FEATURES ONLY ~~~~~")
GaussianNB_model( ae_features_file, '5.Ten_sec_annotated_data/patient_0_1_10.csv' , description = "with AE features only")