from imported_files.paths_n_vars import all_features_file, intra_annotated_file, features_file, ae_features_file

rand_state = 54
test_fraction = 0.5
split_criteria = "entropy"

# k of k flod cross validation
k = 5 # change if you want to experiment

# class list
# class_list = ["0" , "1"] # good segn = 0 , corrupted signal = 1

# ~~~~~~~LIBRARIES~~~~~~~
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn import tree

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
    classifier = tree.DecisionTreeClassifier(random_state = rand_state , criterion = split_criteria)
    classifier.fit(x_train_data , y_train_data)
    return classifier

def k_fold_s_crossval(x_train_data , y_train_data , k_value , classifier):
    # k fold cross validation
    rskf = RepeatedStratifiedKFold(n_splits = k_value , n_repeats = k_value , random_state = rand_state)
    result = cross_val_score(classifier , x_train_data , y_train_data , cv = rskf)
    print("Cross validation Accuracy : mean = {} :: std = {}".format(result.mean() , result.std()))

def get_num_leaf_nodes(classifier , x_train_data):
    # Get the leaf node indices
    leaf_node_indices = classifier.apply(x_train_data)

    # Count the unique leaf node indices
    num_leaf_nodes = len(set(leaf_node_indices))

    return num_leaf_nodes

def one_sample_nodes(classifier):
    n_nodes = classifier.tree_.node_count
    children_left = classifier.tree_.children_left
    children_right = classifier.tree_.children_right

    # Initialize arrays for node depth and leaf status
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)

    stack = [(0, -1)]  # Seed with the root node ID and its parent depth

    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if children_left[node_id] != children_right[node_id]:
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            # This is a leaf node
            is_leaves[node_id] = True

    # Count the nodes with only one sample
    single_sample_nodes = sum(1 for node in range(n_nodes) if is_leaves[node] and node_depth[node] == 1)
    return single_sample_nodes

def test_n_results(x_test_data , y_test_data , classifier , x_train_data , description:str = ""):

    
    class_list = np.unique(y_test_data)

    # visualize the tree
    plt.figure(figsize = (32,20),facecolor='w')
    # print(f"The column names {list(x_test_data.columns)}, \n their types = {type(list(x_test_data.columns))}")
    tree.plot_tree(classifier, feature_names = list(x_test_data.columns) , class_names = ["good segn" , "corrupted"], rounded = True ,filled = True,fontsize= 14)
    plt.show()

    # Now TESTING the model and showing the results:
    # confusion matrix 
    test_pred_decision_tree = classifier.predict(x_test_data) # prediction on test data
    confusion_matrix = metrics.confusion_matrix(y_test_data , test_pred_decision_tree) 
    # plotting confusion matrix
    ax = plt.axes()
    sns.set_theme(font_scale=1.3)
    sns.heatmap(confusion_matrix , annot = True , fmt = "g" , ax = ax , cmap = "magma")
    ax.set_title('Confusion Matrix - Decision Tree: ' + description)
    ax.set_xlabel("Predicted label" , fontsize = 15)
    ax.set_xticklabels(class_list)
    ax.set_ylabel("True Label", fontsize = 15)
    ax.set_yticklabels(class_list, rotation = 0)
    plt.show()

    # classification report
    print("\nClassification Report:\n")
    print(classification_report(y_test_data, test_pred_decision_tree))
    print("\nAvg score on test dataset = {}".format(classifier.score(x_test_data , y_test_data)))
    print("\nThe parameters of Decision tree :")
    print(classifier)
    print(f"Tree depth = {classifier.tree_.max_depth}")
    print(f"Number of nodes = {classifier.tree_.node_count}")
    print(f"Number of Leaf nodes = {get_num_leaf_nodes(classifier , x_train_data)}")
    print(f"Number of nodes with only one sample = {one_sample_nodes(classifier)}")

def dt_model( local_features_file, annotated_file : str = ""  , description : str = ""):
    # get the dataset from the files
    features = pd.read_csv(local_features_file)
    labels = pd.read_csv(annotated_file).iloc[0] # this will exract the annotation 2nd row

    if local_features_file == all_features_file  and 'annotation' in features.columns:
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
    test_n_results(x_test , y_test , clf ,x_train ,  description)

# print("\n~~~~~ DECISION TREE:: W/O AE FEATURES ~~~~~")
# dt_model(features_file ,intra_annotated_file ,  "w/o AE features")
# print("\n~~~~~ DECISION TREE:: WITH ALL FEATURES ~~~~~")
# dt_model(all_features_file , description = "with all features")
print("\n~~~~~ RF:: WITH AE FEATURES ONLY ~~~~~")
dt_model( ae_features_file, intra_annotated_file , description = "with AE features only")
    
# #~~~~~~~~~ AFTER REANNOTATION ~~~~~~~~~~
# print("\n~~~~~ RF:: W/O AE FEATURES :: RE ANNOTATION ~~~~~")
# dt_model('6.Features_extracted/features_filtered_1_1_10.csv' , '5.Ten_sec_annotated_data/patient_0_1_10.csv' , description = "Statistical features")
# print("\n~~~~~ RF::All FEATURES :: RE ANNOTATION ~~~~~")
# dt_model('6.Features_extracted/all_features.csv' , '5.Ten_sec_annotated_data/patient_0_1_10.csv' , description = "All features")
# print("\n~~~~~ RF:: WITH AE FEATURES ONLY ~~~~~")
# dt_model( ae_features_file, '5.Ten_sec_annotated_data/patient_0_1_10.csv' , description = "with AE features only")