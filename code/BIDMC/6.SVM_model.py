annotated_csv_path = "code\BIDMC\Annotated_125Hz\\bidmc03m.csv"
features_file_path = "code\BIDMC\\features_125Hz\\bidmc03m.csv"

rand_state = 42
test_fraction = 0.5
kern = 'linear'
gama = 'auto'
c = 2

# k of k flod cross validation
k = 9 # change if you want to experiment

# class list
class_list = ["1" , "2"] # good signal = 1 , corrupted signal = 2

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

from sklearn.metrics import classification_report
# ~~~~~~~~~~~~END LIBRARIES~~~~~~~~~~~~~~~~~~~~~

def create_train_classifier(X_train_data, y_train_data):
    # fit the SVM model into the train set
    classifier = svm.SVC(kernel = kern, gamma = gama, C = c)
    classifier.fit(X_train_data, y_train_data)
    return classifier

def k_fold_s_crossval(X_train_data, y_train_data, k, classifier):
    # k fold cross validation
    rskf = RepeatedStratifiedKFold(n_splits = k, n_repeats = k, random_state = rand_state)
    result = cross_val_score(classifier, X_train_data, y_train_data, cv = rskf)
    print(result)
    print("Avg accuracy on train set: {}".format(result.mean()))


def test_n_results(X_test_data , y_test_data , classifier):

    # Now TESTING the model and showing the results:
    # confusion matrix 
    test_pred_svm = classifier.predict(X_test_data)
    confusion_matrix = metrics.confusion_matrix(y_test_data , test_pred_svm)
    matrix_df = pd.DataFrame(confusion_matrix)
    ax = plt.axes()
    sns.set(font_scale = 1.3)
    sns.heatmap(matrix_df, annot = True, fmt = "g", ax = ax, cmap = "magma")
    ax.set_title('Confusion Matrix - Support Vector Machine')
    ax.set_xlabel("Predicted label", fontsize = 15)
    ax.set_xticklabels(class_list)
    ax.set_ylabel("True Label", fontsize = 15)
    ax.set_yticklabels(class_list, rotation = 0)
    plt.show()

    # classification report
    print("\nClassification Report:\n")
    print(classification_report(y_test_data, test_pred_svm))
    print("\nAvg score on test dataset = {}".format(classifier.score(X_test_data , y_test_data)))
    # print("\nNumber of Estimators = {}".format(num_trees))
    print("C = {}".format(c))

def svm_model(annotated_file , features_file):
    # get the dataset from the files
    features = pd.read_csv(features_file)
    annotated_data = pd.read_csv(annotated_file)
    labels = annotated_data.iloc[0] # this will exract the annotation 2nd row
    features = features.iloc[:, 1:]

    # split the dataset using test_train_split() function
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = test_fraction, random_state = rand_state, stratify = labels)

    clf = create_train_classifier(X_train , y_train)
    k_fold_s_crossval(X_train , y_train , k , clf)
    test_n_results(X_test , y_test , clf)
    print(clf)


svm_model(annotated_csv_path , features_file_path)

