# ~~~~~~~LIBRARIES~~~~~~~
from numpy import unique
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# for cross validation
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score

# to plot confusion matrix
import matplotlib.pyplot as plt
from seaborn import set_theme , heatmap # seaborn for additional functionality such as heat map

# additionals
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import RocCurveDisplay

# ~~~~~~~~~~~~END LIBRARIES~~~~~~~~~~~~~~~~~~~~~

import functools
def _pass_to_function(function):
        @functools.wraps(function)
        def _pass_to_function_wrapper(*arg , **kwargs):
            function_output = function(*arg , **kwargs)        
            return function_output    
        return _pass_to_function_wrapper

RandomForestClassifier = _pass_to_function(RandomForestClassifier)
DecisionTreeClassifier = _pass_to_function(DecisionTreeClassifier)
SVC = _pass_to_function(SVC)
GaussianNB = _pass_to_function(GaussianNB)
class Ml_helper():
    '''
    * This class provides unified interface to Random forest, Decision Tree, SVM and Gaussian Naive Bayes
    * Just pass the name and it will create that classifier using sklearn library
    '''
    def __init__(self, _ml_model_name: str, **kwargs_custom):
        self.model_name = _ml_model_name
        if _ml_model_name in ['RF' , 'rf' , 'Random Forest' , 'random forest']:
            self.classifier = RandomForestClassifier(**kwargs_custom)
        elif _ml_model_name in ['DT' , 'dt' , 'Decision Tree' , 'decision tree']:
            self.classifier = DecisionTreeClassifier(**kwargs_custom)
        elif _ml_model_name in ['SVM' , 'svm' , 'Support Vector Machine' , 'support vector machine']:
            self.classifier = SVC(**kwargs_custom)
        elif _ml_model_name in ['GaussianNB' , 'GNB', 'gnb' , 'Gaussian Naive Bayes' , 'gaussian naive bayes']:
            self.classifier = GaussianNB(**kwargs_custom)
        
    def k_fold_strat_crossval(self , x_train_data , y_train_data , k_value, rand_state):
        rskf = RepeatedStratifiedKFold(n_splits = k_value , n_repeats = k_value , random_state = rand_state)
        result = cross_val_score(self.classifier , x_train_data , y_train_data , cv = rskf , verbose = 1,n_jobs = -1)
        print("Cross validation Accuracy : mean = {} :: std = {}".format(result.mean() , result.std()))
        

    def test_n_results(self, x_test_data , y_test_data, description:str = ""):

        # Now TESTING the model and showing the results:
        # confusion matrix 
        self.test_pred = self.classifier.predict(x_test_data)
        self.confu_mtrx= confusion_matrix(y_test_data , self.test_pred)
        class_list = unique(y_test_data)
        ax = plt.axes()
        set_theme(font_scale=1.3)
        heatmap(self.confu_mtrx, annot = True , fmt = "g" , ax = ax , cmap = "magma")
        ax.set_title(f'Confusion Matrix - {self.model_name}: {description}')
        ax.set_xlabel("Predicted label" , fontsize = 15)
        ax.set_xticklabels(class_list)
        ax.set_ylabel("True Label", fontsize = 15)
        ax.set_yticklabels(class_list, rotation = 0)
        plt.show()
        
        if len(class_list) == 2:
            # roc curve plot
            ax = plt.gca()
            rfc_disp = RocCurveDisplay.from_estimator(self.classifier, x_test_data, y_test_data, ax=ax, alpha=0.8)
            plt.show()

        # classification report
        print(f"Confu mtrx = \n{self.confu_mtrx}")
        print("\nClassification Report:\n")
        print(classification_report(y_test_data, self.test_pred))
        print("\nAvg score on test dataset = {}".format(self.classifier.score(x_test_data , y_test_data)))
        print(self.classifier)

        return self.test_pred
    
         
