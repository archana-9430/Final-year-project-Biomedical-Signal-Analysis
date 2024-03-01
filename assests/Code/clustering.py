'''cluster the segments of patients based on their annotations and
then find the Euclidean distances from the cluster centers. '''

from imported_files.paths_n_vars import features_file, all_features_file, intra_annotated_file

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

# before re-annotation
df = pd.read_csv(features_file)
annotations = pd.read_csv(intra_annotated_file).iloc[0].values.astype('int')
print(f'annotations = {annotations}')

# # Drop patient IDs and annotations
# data = df.drop([0])

# # after re-annotation
# df = pd.read_csv(all_features_file)
# annotations = pd.read_csv('5.Ten_sec_annotated_data/patient_0_1_10.csv').iloc[0].T.values.astype(int)
data = df


X = data.astype(float)
print(type(X))
print(X)
# X = (X - X.mean())/X.std()
# X = (X - X.min())/(X.max() - X.min())
print(X)

#Elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(len(wcss)), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#Optimal number of clusters from the elbow method plot
n_clusters = 4

#KMeans clustering
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
kmeans.fit(X)
cluster_centers = kmeans.cluster_centers_
num_features = cluster_centers.shape[1]
print("Number of features in each center point: ", num_features)

#Euclidean distances from cluster centers
distances = pairwise_distances(cluster_centers, cluster_centers)
print("Euclidean Distances of each Cluster Centers: ")
print(distances)
features_df = pd.DataFrame(distances)
features_df = features_df.T
features_df.to_csv("cluster_euclidean_distances.csv", index = False)

#Range of cluster centers along each feature dimension for each cluster -> size == num of clusters
cluster_ranges = []
for cluster_center in cluster_centers:
    cluster_range = cluster_center.max() - cluster_center.min()
    cluster_ranges.append(cluster_range)
cluster_ranges = np.array(cluster_ranges)
print('Range of cluster centers along each feature dimension for each cluster: ', cluster_ranges)

#Standard deviation of cluster centers along each feature dimension -> size == num of features
cluster_centers_std = np.std(cluster_centers, axis=0)
print("Standard deviation of cluster centers along each feature dimension:")
print(cluster_centers_std)

# composition of each cluster
# predict cluster of each segment on the already fitted model
print('\n')
cluster_labels = kmeans.predict(X)
print(f"number of clusters = {n_clusters}")
print(f"cluster labels = {cluster_labels}")
print(f"cluster labels shape = {cluster_labels.shape}")
print(f"annotations = {annotations}")
print(f"annotations shape = {annotations.shape}")
tuple_anno_clus = tuple(zip(annotations , cluster_labels))
print(f'Tuple of annotation and cluster id:\n{tuple_anno_clus}')

composition = {}
for i in range(n_clusters):
    clus_composition = {'0' : 0 , '1' : 0}
    for x , y in tuple_anno_clus:
        if y == i:
            if x:
                clus_composition['1'] += 1
            else:
                clus_composition['0'] += 1
    composition[f'Cluster { i } '] = clus_composition

from pprint import pprint
pprint(composition)
