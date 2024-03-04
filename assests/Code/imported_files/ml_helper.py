from paths_n_vars import features_file, intra_annotated_file, all_features_file, ae_features_file

rand_state = 54
test_fraction = 0.5
num_trees = 15
split_criteria = "entropy"

# k of k fold cross validation
k = 9 # change if you want to experiment

# class list
# class_list = ["0" , "1"] # good segn = 0 , corrupted signal = 1

# ~~~~~~~LIBRARIES~~~~~~~
from pandas import read_csv
from numpy import unique, any, isnan
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# for cross validation
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score

# to plot confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns # seaborn for additional functionality such as heat map

# additionals
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import RocCurveDisplay

# to calculate number of instances of 0 and 1
from collections import Counter
# ~~~~~~~~~~~~END LIBRARIES~~~~~~~~~~~~~~~~~~~~~

    
def k_fold_s_crossval(x_train_data , y_train_data , k_value , classifier):
    rskf = RepeatedStratifiedKFold(n_splits = k_value , n_repeats = k_value , random_state = rand_state)
    result = cross_val_score(classifier , x_train_data , y_train_data , cv = rskf , verbose = 1)
    print("Cross validation Accuracy : mean = {} :: std = {}".format(result.mean() , result.std()))

def rf_model( local_features_file, annotated_file , description : str = ""):
    # get the dataset from the files
    features = read_csv(local_features_file)
    labels = read_csv(annotated_file).iloc[0] # this will exract the annotation 2nd row

    if local_features_file == annotated_file:
        features.drop(index=0, inplace=True)
        features = features.T
    if 'annotation' in features.columns :
        features.drop(['annotation'] , axis = 'columns' , inplace = True)

    assert not any(isnan(features)) , "ERROR::FEATURES DATAFRAME HAS NAN VALUES"
    # split the dataset using test_train_split() function
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size = test_fraction, random_state = rand_state, stratify = labels)

    num_instances_train = dict(Counter(y_train))
    num_instances_test = dict(Counter(y_test))
    print(f"Train instances : {num_instances_train}")
    print(f"Test instances : {num_instances_test}")

    clf = create_train_classifier(x_train , y_train)
    k_fold_s_crossval(x_train , y_train , k , clf)
    test_n_results(x_test , y_test , clf , description)

class Ml_Model():
    def __init__(self, ml_model_name: str, rand_state : int = 54 ):
        self.model_name = ml_model_name
        self.classifier = self.create_classifier(ml_model_name)

    def create_classifier(self, ml_model_name:str):
        return RandomForestClassifier(n_estimators = num_trees , random_state = rand_state \
                                  , criterion = split_criteria, verbose = 1)

    def test_n_results(self, x_test_data , y_test_data, description:str = ""):

        # Now TESTING the model and showing the results:
        # confusion matrix 
        test_pred = self.classifier.predict(x_test_data)
        confu_mtrx= confusion_matrix(y_test_data , test_pred)
        class_list = unique(y_test_data)
        ax = plt.axes()
        sns.set_theme(font_scale=1.3)
        sns.heatmap(confu_mtrx, annot = True , fmt = "g" , ax = ax , cmap = "magma")
        ax.set_title(f'Confusion Matrix - {self.model_name}: {description}')
        ax.set_xlabel("Predicted label" , fontsize = 15)
        ax.set_xticklabels(class_list)
        ax.set_ylabel("True Label", fontsize = 15)
        ax.set_yticklabels(class_list, rotation = 0)
        plt.show()

        # # roc curve plot
        # ax = plt.gca()
        # rfc_disp = metrics.RocCurveDisplay.from_estimator(self.classifier, x_test_data, y_test_data, ax=ax, alpha=0.8)
        # plt.show()

        # classification report
        print(f"Confu mtrx = \n{confusion_matrix}")
        print("\nClassification Report:\n")
        print(classification_report(y_test_data, test_pred))
        print("\nAvg score on test dataset = {}".format(self.classifier.score(x_test_data , y_test_data)))
        print(self.classifier)
