import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from imported_files.paths_n_vars import all_features_file , features_file, intra_annotated_file

def MyZScoreScaler(dataframe):
    '''
    Scales each PPG segment individually
    Function assumes the PPG segments are present column wise in the given DataFrame
    '''
    return (dataframe - dataframe.mean())/dataframe.std()

import time
import functools
def custom_timeit(function):
    @functools.wraps(function)
    def wrapper_custom_timeit(*arg , **kwargs):
        
        start = time.perf_counter()
        function_output = function(*arg , **kwargs)
        end = time.perf_counter()
        print(f"\nPerformance :: {function.__name__} took : {end - start : 0.8f} seconds\n")
        
        return function_output
    
    return wrapper_custom_timeit

from matplotlib.pyplot import show
def wrapper_scatter_plot(function):
    @functools.wraps(function)
    def wrapper_of_wrapper(*arg , **kwargs):
        ax = function(*arg , **kwargs)
        show()
        return ax
    return wrapper_of_wrapper

def tSNE_visualization(path):
    dataframe = pd.read_csv(path)
    print(f"Feature matrix dimensions = {dataframe.shape}")
    tSNE = TSNE(n_components = 3 , verbose = 1 , random_state = 2)
    annotations = pd.DataFrame()
    if 'annotation' in dataframe.columns:
        print(f"file name = {path}")
        annotations = dataframe['annotation']
        dataframe.drop(['annotation'] , axis = 1 , inplace = True)
    else:
        annotations = pd.read_csv(intra_annotated_file).iloc[ 0 ]

    print(f"annotation of file : {path} are:\n{annotations}")

    tSNE.fit_transform = custom_timeit(tSNE.fit_transform)
    reduced_components = tSNE.fit_transform(dataframe)
    df_tSNE = pd.DataFrame()
    df_tSNE['component - 1'] = reduced_components[ : , 0]
    df_tSNE['component - 2'] = reduced_components[ : , 1]
    df_tSNE['component - 3'] = reduced_components[ : , 2]

    fig = px.scatter_3d(x= df_tSNE['component - 1'], y= df_tSNE['component - 2'], z= df_tSNE['component - 3'], color=annotations, opacity=1)
    fig.show()

    # Applying scatter plot decorator
    # sns.scatterplot = wrapper_scatter_plot(sns.scatterplot)

    # sns.scatterplot(x = "component - 1", y = "component - 2", hue = annotations.tolist(),
    #                 palette = sns.color_palette("deep", 2),
    #                 data = df_tSNE).set(title=f"T-SNE projection of : {path}")
    
# tSNE_visualization(features_file)
# tSNE_visualization(all_features_file)

#%%
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imported_files.paths_n_vars import features_file

def visualization(path):
    dataframe = pd.read_csv(path)
    print(f"Feature matrix dimensions = {dataframe.shape}")
    dataframe = MyZScoreScaler(dataframe)
    if 'annotation' in dataframe.columns:
        print(f"file name = {path}")
        dataframe.drop(['annotation'] , axis = 1 , inplace = True)

    correlation_matrix = np.corrcoef(dataframe)

    sns.heatmap(correlation_matrix , annot = True , cmap = 'coolwarm')
    plt.title(f'Heatmap of {path}')
    plt.show()

visualization(features_file)


#%%
# def feature_visualization(path):
#     dataframe = pd.read_csv(path)
#     print(f"Feature matrix dimensions = {dataframe.shape}")
#     # tSNE = TSNE(n_components = 2 , verbose = 1 , random_state = 2)
#     annotations = pd.DataFrame()
#     if 'annotation' in dataframe.columns:
#         print(f"file name = {path}")
#         annotations = dataframe['annotation']
#         dataframe.drop(['annotation'] , axis = 1 , inplace = True)
#     else:
#         annotations = pd.read_csv(intra_annotated_file).iloc[ 0 ]

#     df_norm = MyZScoreScaler(dataframe)
#     print(f"annotation of file : {path} are:\n{annotations}")

#     df_features = pd.DataFrame()
#     df_features['component - 1'] = df_norm.iloc[ : , 10]
#     df_features['component - 2'] = df_norm.iloc[ : , 8]
#     # df_features['component - 3'] = reduced_components[ : , 2]

#     # fig = px.scatter_3d(x= df_features['component - 1'], y= df_features['component - 2'], z= df_features['component - 3'], color=annotations, opacity=1)
#     # fig.show()

#     # Applying scatter plot decorator
#     sns.scatterplot = wrapper_scatter_plot(sns.scatterplot)

#     sns.scatterplot(x = "component - 1", y = "component - 2", hue = annotations.tolist(),
#                     palette = sns.color_palette("deep", 2),
#                     data = df_features).set(title=f"T-SNE projection of : {path}")
    

# feature_visualization(features_file)

#%% 
# import seaborn
 
 
# seaborn.set(style='whitegrid')
# fmri = seaborn.load_dataset("fmri")
 
# ax = plt.axes()
# ax =  (seaborn.scatterplot(x="timepoint",
#                     y="signal",
#                     data=fmri))
# plt.show()


# from sklearn.manifold import TSNE
# from keras.datasets import mnist
# from sklearn.datasets import load_iris
# from numpy import reshape
# import seaborn as sns
# import pandas as pd

# sns.scatterplot = wrapper_scatter_plot(sns.scatterplot)

# iris = load_iris()
# x = iris.data
# y = iris.target

# tsne = TSNE(n_components=2, verbose=1, random_state=123)
# z = tsne.fit_transform(x)
# df = pd.DataFrame()
# df["y"] = y
# df["comp-1"] = z[:,0]
# df["comp-2"] = z[:,1]

# sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
#                 palette=sns.color_palette("hls", 3),
#                 data=df).set(title="Iris data T-SNE projection")

# (x_train, y_train), (_ , _) = mnist.load_data()
# x_train = x_train[:3000]
# y_train = y_train[:3000]
# print(x_train.shape) 
 
# x_mnist = reshape(x_train, [x_train.shape[0], x_train.shape[1]*x_train.shape[2]])
# print(x_mnist.shape)

# tsne = TSNE(n_components=2, verbose=1, random_state=123)
# z = tsne.fit_transform(x_mnist)
 
# df = pd.DataFrame()
# df["y"] = y_train
# df["comp-1"] = z[:,0]
# df["comp-2"] = z[:,1]

# sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
#                 palette=sns.color_palette("hls", 10),
#                 data=df).set(title="MNIST data T-SNE projection")