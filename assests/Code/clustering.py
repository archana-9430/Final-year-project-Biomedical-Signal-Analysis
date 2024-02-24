'''cluster the segments of patients based on their annotations and
then find the Euclidean distances from the cluster centers. '''

from imported_files.paths_n_vars import intra_annotated_file, features_file

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

# df = pd.read_csv(intra_annotated_file)
df = pd.read_csv(features_file)
annotations = df.iloc[1].values

# Drop patient IDs and annotations
data = df.drop([0])
X = data.astype(float)
print(type(X))
print(X)

#Elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
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
print("Number of features in each center point:", num_features)

#Euclidean distances from cluster centers
distances = pairwise_distances(cluster_centers, cluster_centers)
print("Euclidean Distances of each Cluster Centers: ")
print(distances)
features_df = pd.DataFrame(distances)
features_df = features_df.T
features_df.to_csv("cluster_euclidean_distances.csv", index = False)
