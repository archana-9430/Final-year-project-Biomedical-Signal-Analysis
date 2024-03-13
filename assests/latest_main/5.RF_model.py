from imported_files.paths_n_vars import stats_features, filtered_merged, feature_merged
from imported_files.ml_helper import Ml_Model

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
from sklearn.model_selection import train_test_split

# to calculate number of instances of 0 and 1
from collections import Counter
# ~~~~~~~~~~~~END LIBRARIES~~~~~~~~~~~~~~~~~~~~~

def rf_model_function( local_features_file, annotated_file , description : str = ""):
    # get the dataset from the files
    features = pd.read_csv(local_features_file)
    labels = pd.read_csv(annotated_file).iloc[0] # this will exract the annotation 2nd row    
    class_list = np.unique(labels)

    if local_features_file == annotated_file:
        features.drop(index=0, inplace=True)
        features = features.T
        
    if 'annotation' in features.columns :
        features.drop(['annotation'] , axis = 'columns' , inplace = True)

    features_train, features_test, y_train, y_test = train_test_split(features, labels, test_size = test_fraction, random_state = rand_state, stratify = labels)

    if 'PatientID' in features.columns:
        pID_train = features_train['PatientID']
        pID_test = features_test['PatientID']
        x_train = features_train.drop(['PatientID'] , axis = 'columns')
        x_test = features_test.drop(['PatientID'] , axis = 'columns')

    else:
        x_train = features_train
        x_test = features_test

    print("Number of features: ", x_train.shape[1])
    num_instances_train = dict(Counter(y_train))
    num_instances_test = dict(Counter(y_test))
    print(f"Train instances : {num_instances_train}")
    print(f"Test instances : {num_instances_test}")

    # create and train classifier
    rf_model = Ml_Model('rf' , n_estimators = num_trees , random_state = rand_state \
                                  , criterion = split_criteria, verbose = 1)
    rf_model.classifier.fit(x_train , y_train)
    rf_model.k_fold_strat_crossval(x_train , y_train , k , rand_state)
    y_pred = rf_model.test_n_results(x_test , y_test , description)
    
    if 'PatientID' in features.columns:
        # identifying miss-classisfied segments and saving them
        classification_rep = tuple(zip(pID_test.values , y_test.values , y_pred))
        num_patients = np.unique( [x.split('_')[0] for x in pID_test.values])
        print('number of patients = ',num_patients)
        dataset = pd.read_csv('annotated_merged.csv')

        dataset[[x[0] for x in classification_rep if x[2] == 0] ].to_csv(path_or_buf = 'Stage_2/clean.csv',index = False)
        dataset[[x[0] for x in classification_rep if x[2] == 1] ].to_csv(path_or_buf = 'Stage_2/partly_corrupted.csv',index = False)
        dataset[[x[0] for x in classification_rep if x[2] == 2] ].to_csv(path_or_buf = 'Stage_2/corrupted.csv',index = False)


        allMissClassifications = [x for x in classification_rep if x[1] != x[2]] # x[1] -> true labels :: x[2] -> predicted labels
        
        
        missClasslist = []
        for i in range(len(class_list)):
            innerList = []
            for j in range(len(class_list)):
                temp = [x[0] for x in allMissClassifications if ((x[1] == i) and (x[2] == j))] 
                if len(temp) > 0:
                    innerList.append(temp)
                    dataset[temp].to_csv(path_or_buf = f'4.missclassifications/missclassification_Of{i}_to_{j}.csv',index = False)
            if len(innerList) > 0:
                missClasslist.append(innerList)
                
    else:
        print('Patient ID not given so cannot identify missclassified patients...')
    
print("\n~~~~~ RF:: W/O AE FEATURES ~~~~~")
rf_model_function(stats_features, filtered_merged, description = "Statistical features")

print("\n~~~~~ RF:: WITH ALL FEATURES ~~~~~")
rf_model_function(feature_merged, filtered_merged, description = "All features")