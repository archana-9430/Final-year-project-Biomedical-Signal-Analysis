# evaluate RFE for classification
from numpy import array
from pandas import read_csv
from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from imported_files.paths_n_vars import features_file , intra_annotated_file , all_features_file


def MyMinMaxScaler(dataframe):
    return (dataframe - dataframe.min())/(dataframe.max() - dataframe.min())

def MyZScoreScaler(dataframe):
    return (dataframe - dataframe.mean())/dataframe.std()
n = 0
# get the dataset from the files
features = read_csv(features_file)
labels = read_csv(intra_annotated_file).iloc[0] # this will exract the annotation 2nd row

if features_file == all_features_file:
    features.drop(['annotation'] , axis = 'columns' , inplace = True)

if(n == 1):
    features = MyMinMaxScaler(features)
if(n == 2):
    features = MyZScoreScaler(features)


# create pipeline
rfe = RFE(estimator = RandomForestClassifier(), n_features_to_select = 5)
model = RandomForestClassifier( n_estimators = 10 , criterion = 'entropy')
pipeline = Pipeline(steps=[('s',rfe),('m',model)])
# evaluate model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(pipeline , features , labels , scoring='accuracy' , cv = cv , n_jobs = -1 , error_score = 'raise')
# report performance
print('Accuracy: %.5f (%.5f)' % (mean(n_scores), std(n_scores)))
support = pipeline.named_steps['s'].support_
feature_names = array(features.columns) # transformed list to array
print(array(feature_names)[support])
