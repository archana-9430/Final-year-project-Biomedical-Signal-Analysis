
import sys
IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
  from google.colab import drive
  drive.mount('/content/drive' , force_remount = True)
  intra_annotated_file = "/content/drive/MyDrive/B_Tech_ECE/Own Final Year Project/combined_annotated.csv"
  ae_features_file = "/content/drive/MyDrive/B_Tech_ECE/Own Final Year Project/AE_features.csv"
  encoder_save = "/content/drive/MyDrive/B_Tech_ECE/Own Final Year Project/encoder_model.keras"
  decoder_save = "/content/drive/MyDrive/B_Tech_ECE/Own Final Year Project/decoder_model.keras"
else:
  from imported_files.paths_n_vars import intra_annotated_file

# Use scikit-learn to grid search the batch size and epochs
import pandas as pd
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, train_test_split
from tensorflow.keras.models import load_model
from scikeras.wrappers import KerasClassifier

# Function to create model, required for KerasClassifier
def create_model():
    # create model
    encoder = load_model('encoder_model.keras')
    decoder = load_model('decoder_model.keras')
    # return model without compile
    return  decoder(encoder)

def MyMinMaxScaler(dataframe : pd.DataFrame):
    '''
    Scales each PPG segment individually
    Function assumes the PPG segments are present column wise in the given DataFrame
    '''
    return (dataframe - dataframe.min())/(dataframe.max() - dataframe.min())

# load dataset
annotated_data = pd.read_csv(intra_annotated_file )
# annotated_data = pd.read_csv('5.New_annotated_data\patient_0_1_10.csv' )
raw_segments = (annotated_data).iloc[1 : , ]
labels = annotated_data.iloc[0] # this will exract the annotation 2nd row
dataset = MyMinMaxScaler(raw_segments).values.T # mind the Transpose operation

rand_state = 54
test_fraction = 0.3
num_epochs = 300
batch_size = 32
k = 2

# split into input (X) and output (Y) variables
x_train, x_test, y_train, y_test = train_test_split(dataset , labels, test_size = test_fraction, random_state = rand_state, stratify = labels)

# create model
model = KerasClassifier(model=create_model, loss="mean_squared_error", epochs=num_epochs, batch_size=batch_size, verbose=0)
# define the grid search parameters
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
batch_size = [x for x in range(2,40)]
param_grid = dict(optimizer=optimizer , batch_size = batch_size)
rskf = RepeatedStratifiedKFold(n_splits = k , n_repeats = k , random_state = rand_state)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=rskf)
grid_result = grid.fit(x_train, x_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))