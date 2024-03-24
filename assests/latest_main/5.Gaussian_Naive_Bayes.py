from imported_files.paths_n_vars import stats_features, filtered_merged, feature_merged
from imported_files.ml_helper import Ml_helper

rand_state = 54
test_fraction = 0.5

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

def GaussianNB_model_function( local_features_file , description : str = ""):
    # get the dataset from the files
    features_df = pd.read_csv(local_features_file)
    labels = features_df['annotation'] # this will exract the annotation 2nd row    
    class_list = np.unique(labels)
        
    if 'annotation' in features_df.columns :
        features_df.drop(['annotation'] , axis = 'columns' , inplace = True)

    assert not 'annotation' in features_df.columns

    features_train, features_test, y_train, y_test = train_test_split(features_df, labels, test_size = test_fraction, random_state = rand_state, stratify = labels)

    if 'PatientID' in features_df.columns:
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

    assert not ('annotation' in x_train.columns or 'PatientID' in x_train.columns)
    # create and train classifier
    GaussianNB_model = Ml_helper('GaussianNB')
    GaussianNB_model.classifier.fit(x_train , y_train)
    GaussianNB_model.k_fold_strat_crossval(x_train , y_train , k , rand_state)
    y_pred = GaussianNB_model.test_n_results(x_test , y_test , description)
    
    allMissClassifications = []
    missClasslist = []
    if 'PatientID' in features_df.columns:
        # identifying miss-classisfied segments and saving them
        classification_rep = tuple(zip(pID_test.values , y_test.values , y_pred))
        unique_patients = np.unique( [x.split('_')[0] for x in pID_test.values])
        print('number of patients = ',len(unique_patients))
        dataset = pd.read_csv('annotated_merged.csv')

        dataset[[x[0] for x in classification_rep if x[2] == 0] ].to_csv(path_or_buf = 'Stage_2/GNB/clean.csv',index = False)
        dataset[[x[0] for x in classification_rep if x[2] == 1] ].to_csv(path_or_buf = 'Stage_2/GNB/partly_corrupted.csv',index = False)
        dataset[[x[0] for x in classification_rep if x[2] == 2] ].to_csv(path_or_buf = 'Stage_2/GNB/corrupted.csv',index = False)


        allMissClassifications = [x for x in classification_rep if x[1] != x[2]] # x[1] -> true labels :: x[2] -> predicted labels
        
        for i in range(len(class_list)):
            innerList = []
            for j in range(len(class_list)):
                temp = [x[0] for x in allMissClassifications if ((x[1] == i) and (x[2] == j))] 
                if len(temp) > 0:
                    innerList.append(temp)
                    dataset[temp].to_csv(path_or_buf = f'4.missclassifications/GNB/{description}_{i}_to_{j}.csv',index = False)
            missClasslist.append(innerList)
                
    else:
        print('Patient ID not given so cannot identify missclassified patients...')
        missClasslist = None

    return np.array(allMissClassifications) , missClasslist
    
print("\n~~~~~ GaussianNB:: W/O AE FEATURES ~~~~~")
list_stats , stat_miss_class = GaussianNB_model_function(stats_features, description = "Statistical features")

print("\n~~~~~ GaussianNB:: WITH ALL FEATURES ~~~~~")
list_all , all_miss_class = GaussianNB_model_function(feature_merged, description = "All features")
# for i in range(len(list_all)):
#     if len(list_all[i]):
#         for x in list_all[i]:
#             if x not in list_stats[i]:
#                 print(x)
only_ae = []
for x in list_all:
    if x[0] not in list_stats[:,0]:
        only_ae.append(x.tolist())
    
# print(f'{only_ae = }')
print(f'{list_stats.shape = }')
print(f'{list_all.shape = }')
print(f'{len(only_ae) = }')
