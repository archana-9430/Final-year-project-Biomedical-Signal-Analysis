from imported_files.paths_n_vars import features_file, intra_annotated_file, all_features_file

rand_state = 42
test_fraction = 0.5
kern = 'linear'
gama = 'auto'
c = 2

# k of k flod cross validation
k = 9 # change if you want to experiment

# class list
# class_list = ["0" , "1"] # good signal = 1 , corrupted signal = 2

# ~~~~~~~LIBRARIES~~~~~~~
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm

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

def create_train_classifier(x_train_data, y_train_data):
    # fit the SVM model into the train set
    classifier = svm.SVC(kernel = kern, gamma = gama, C = c, verbose = True)
    classifier.fit(x_train_data, y_train_data )
    return classifier

def k_fold_s_crossval(x_train_data, y_train_data, k, classifier):
    # k-fold cross validation
    rskf = RepeatedStratifiedKFold(n_splits = k, n_repeats = k, random_state = rand_state)
    result = cross_val_score(classifier, x_train_data, y_train_data, cv = rskf)
    print(result)
    print("Avg accuracy on train set: {}".format(result.mean()))


def test_n_results(x_test_data , y_test_data , classifier , description:str = ""):

    # Now TESTING the model and showing the results using confusion matrix 
    test_pred_svm = classifier.predict(x_test_data)
    confusion_matrix = metrics.confusion_matrix(y_test_data , test_pred_svm)
    class_list = np.unique(y_test_data)
    ax = plt.axes()
    sns.set_theme(font_scale=1.3)
    sns.heatmap(confusion_matrix , annot = True , fmt = "g" , ax = ax , cmap = "magma")
    ax.set_title('Confusion Matrix - Support Vector Machines n: ' + description)
    ax.set_xlabel("Predicted label" , fontsize = 15)
    ax.set_xticklabels(class_list)
    ax.set_ylabel("True Label", fontsize = 15)
    ax.set_yticklabels(class_list, rotation = 0)
    plt.show() 

    # classification report
    print("\nClassification Report:\n")
    print(classification_report(y_test_data, test_pred_svm))
    print("\nAvg score on test dataset = {}".format(classifier.score(x_test_data , y_test_data)))
    print(classifier)

def svm_model(local_features_file, annotated_file : str = ""  , description : str = ""):
    # get the dataset from the files
    features = pd.read_csv(local_features_file)
    labels = pd.read_csv(annotated_file).iloc[0] # this will exract the annotation 2nd row

    if local_features_file == all_features_file and 'annotation' in features.columns :
        features.drop(['annotation'] , axis = 'columns' , inplace = True)

    assert not np.any(np.isnan(features)) , "ERROR::FEATURES DATAFRAME HAS NAN VALUES"

    # split the dataset using test_train_split() function
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size = test_fraction, random_state = rand_state, stratify = labels)

    # print("x_train , y_train", x_train , y_train)

    num_instances_train = dict(Counter(y_train))
    num_instances_test = dict(Counter(y_test))
    print(f"Train instances : {num_instances_train}")
    print(f"Test instances : {num_instances_test}")

    clf = create_train_classifier(x_train , y_train)
    k_fold_s_crossval(x_train , y_train , k , clf)
    test_n_results(x_test , y_test , clf , description)

# print("\n~~~~~ SVM:: W/O AE FEATURES ~~~~~")
# svm_model(features_file , intra_annotated_file , description = "w/o AE features")
# print("\n~~~~~ SVM:: WITH ALL FEATURES ~~~~~")
# svm_model(all_features_file , description = "with all features")

#~~~~~~~~~ AFTER REANNOTATION ~~~~~~~~~~
print("\n~~~~~ RF:: W/O AE FEATURES :: RE ANNOTATION ~~~~~")
svm_model('6.Features_extracted/features_filtered_1_1_10.csv' , '5.Ten_sec_annotated_data/patient_0_1_10.csv' , description = "Statistical features")
print("\n~~~~~ RF::All FEATURES :: RE ANNOTATION ~~~~~")
svm_model('6.Features_extracted/all_features.csv' , '5.Ten_sec_annotated_data/patient_0_1_10.csv' , description = "All features")