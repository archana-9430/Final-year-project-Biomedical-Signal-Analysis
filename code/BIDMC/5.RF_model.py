annotated_csv_path = "code\BIDMC\Annotated_125Hz\\bidmc03m.csv"
features_file_path = "code\BIDMC\\features_125Hz\\bidmc03m.csv"

rand_state = 54
test_fraction = 0.5
num_trees = 5
split_criteria = "gini"

# k of k flod cross validation
k = 4 # change if you want to experiment

# class list
class_list = ["1" , "2"] # good segn = 1 , corrupted signal = 2

# ~~~~~~~LIBRARIES~~~~~~~
import pandas as pd
import numpy as np
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

from sklearn.metrics import classification_report
# ~~~~~~~~~~~~END LIBRARIES~~~~~~~~~~~~~~~~~~~~~

def create_train_classifier(x_train_data , y_train_data):
    # fit the decision tree model into the train set
    classifier = RandomForestClassifier(n_estimators = num_trees , random_state = rand_state , criterion = split_criteria)
    classifier.fit(x_train_data , y_train_data)
    return classifier

def k_fold_s_crossval(x_train_data , y_train_data , k , classifier):
    # k fold cross validation
    rskf = RepeatedStratifiedKFold(n_splits = k , n_repeats = k , random_state = rand_state)
    result = cross_val_score(classifier , x_train_data , y_train_data , cv = rskf)
    print(result)
    print("Avg accuracy on train set: {}".format(result.mean()))


def test_n_results(x_test_data , y_test_data , classifier):

    # Now TESTING the model and showing the results:
    # confusion matrix 
    test_pred_decision_tree = classifier.predict(x_test_data)
    confusion_matrix = metrics.confusion_matrix(y_test_data , test_pred_decision_tree)
    matrix_df = pd.DataFrame(confusion_matrix)
    ax = plt.axes()
    sns.set(font_scale=1.3)
    sns.heatmap(matrix_df , annot = True , fmt = "g" , ax = ax , cmap = "magma")
    ax.set_title('Confusion Matrix - Random Forest')
    ax.set_xlabel("Predicted label" , fontsize = 15)
    ax.set_xticklabels(class_list)
    ax.set_ylabel("True Label", fontsize = 15)
    ax.set_yticklabels(class_list, rotation = 0)
    plt.show()

    # classification report
    print("\nClassification Report:\n")
    print(classification_report(y_test_data, test_pred_decision_tree))
    print("\nAvg score on test dataset = {}".format(classifier.score(x_test_data , y_test_data)))
    print("\nNumber of Estimators = {}".format(num_trees))
    print("Split Criteria = {}".format(split_criteria))
    print(classifier)

def RF_model(annotated_file , features_file):
    # get the dataset from the files
    features = pd.read_csv(features_file)
    annotated_data = pd.read_csv(annotated_file)
    labels = annotated_data.iloc[0] # this will exract the annotation 2nd row
    

    # split the dataset using test_train_split() function
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size = test_fraction, random_state = rand_state, stratify = labels)

    clf = create_train_classifier(x_train , y_train)
    k_fold_s_crossval(x_train , y_train , k , clf)
    test_n_results(x_test , y_test , clf)


RF_model(annotated_csv_path , features_file_path)

