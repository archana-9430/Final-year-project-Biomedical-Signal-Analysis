from imported_files.paths_n_vars import stats_features, filtered_merged, feature_merged, AE_features, annotated_merged
from imported_files.ml_helper import Ml_helper
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

rand_state = 54
test_fraction = 0.5
num_trees = 50
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
def save_model_predictions(fil_mer_path , classification_rep, unique_patients):
    filtered_m_df = pd.read_csv(fil_mer_path)
    for u_p in unique_patients:
        l1 = [x[0] for x in classification_rep if x[2] == 0 and x[0].split('_')[0] == u_p]
        l2 = [x[0] for x in classification_rep if x[2] == 1 and x[0].split('_')[0] == u_p]
        l3 = [x[0] for x in classification_rep if x[2] == 2 and x[0].split('_')[0] == u_p]
        filtered_m_df[l1].to_csv(path_or_buf = 'Stage_2/rf/model_prediction/clean/patient_'+ u_p +'.csv',index = False)        
        filtered_m_df[l2].to_csv(path_or_buf = 'Stage_2/rf/model_prediction/partly_corrupted/patient_'+ u_p +'.csv',index = False)
        filtered_m_df[l3].to_csv(path_or_buf = 'Stage_2/rf/model_prediction/corrupted/patient_'+ u_p +'.csv',index = False)

def identify_missclass(anno_mer_path, classification_report, unique_patients, description):
    missClasslist = []
    allMissClassifications = []
    allMissClassifications = [x for x in classification_report if x[1] != x[2]] # x[1] -> true labels :: x[2] -> predicted label
    dataset = pd.read_csv(anno_mer_path)
    for i in range(len(unique_patients)):
        innerList = []
        for j in range(len(unique_patients)):
            temp = [x[0] for x in allMissClassifications if ((x[1] == i) and (x[2] == j))] 
            if len(temp) > 0:
                innerList.append(temp)
                dataset[temp].to_csv(path_or_buf = f'4.missclassifications/rf/{description}_{i}_to_{j}.csv',index = False)
        missClasslist.append(innerList)
    return np.array(allMissClassifications) , missClasslist

def classification_report_(fil_mer_path, anno_mer_path, classification_report, description):
    
    unique_patients = np.unique( [x[0].split('_')[0] for x in classification_report])
    # save model's predictons
    print('_'*50)
    print("SAVING MODEL'S PREDICTION")
    save_model_predictions(fil_mer_path , classification_report , unique_patients)

    
    print('_'*50)
    print('IDENTIFYING MISSCLASSIFICATIONS')
    print('number of patients = ',len(unique_patients))

    return identify_missclass(anno_mer_path, classification_report, unique_patients, description)
    

def rf_model_function( local_features_file , description : str = ""):
    # get the dataset from the files
    features_df = pd.read_csv(local_features_file)
    labels = features_df['annotation'] # this will exract the annotation 2nd row    
    unanno_df = features_df.drop(['annotation'] , axis = 'columns')

    assert not 'annotation' in unanno_df.columns

    features_train, features_test, y_train, y_test = train_test_split(unanno_df, labels, test_size = test_fraction, random_state = rand_state, stratify = labels)

    if 'PatientID' in features_df.columns:
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
    rf_model = Ml_helper('rf' , n_estimators = num_trees , random_state = rand_state \
                                  , criterion = split_criteria, n_jobs = -1)
    
    # # hyper parameter tuning
    # print('Hyper parameter tuning starts')    
    # param_grid = { 
    # 'n_estimators': range(5,151,5),
    # 'max_features': ['sqrt', 'log2'],
    # 'criterion': ['gini','entropy'],
    # 'max_depth':range(5,21,5)
    # }
    # cv_rf = GridSearchCV(rf_model.classifier, param_grid, n_jobs=-1, cv=5, refit=True)
    # cv_rf.fit(x_train,y_train)
    # print(cv_rf.best_params_)
    # rf_model.classifier = cv_rf.best_estimator_
    # # tuning ends

    rf_model.classifier.fit(x_train , y_train)
    rf_model.k_fold_strat_crossval(x_train , y_train , k , rand_state)
    y_pred = rf_model.test_n_results(x_test , y_test , description)
    
    
    if 'PatientID' in features_df.columns:
        classification_rep = tuple(zip(features_test['PatientID'].values , y_test.values , y_pred))
        return classification_report_(filtered_merged, annotated_merged, classification_rep, description)
    
    print('Patient ID not given so cannot identify missclassified patients...')
    return None, None

print("~"*25 + " RF:: W/O AE FEATURES " + "~"*25)
list_stats , stat_miss_class = rf_model_function(stats_features, description = "Statistical features")

print("~"*25 + " RF:: ONLY AE FEATURES " + "~"*25)
list_ae , ae_miss_class = rf_model_function(AE_features, description = "AE features")

print("~"*25 + " RF:: WITH ALL FEATURES " + "~"*25)
list_all , all_miss_class = rf_model_function(feature_merged, description = "All features")


