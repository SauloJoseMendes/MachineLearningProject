import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from subtractive_clustering.subtractive_main import subtractive_clustering, assign_points_to_clusters
def plotCluster(title):
    plt.figure(figsize=(8, 6))
    plt.scatter(data_for_clustering.iloc[:, 0], data_for_clustering.iloc[:, 1], c=numeric_data['Cluster'],
                cmap='viridis', s=50, alpha=0.7)
    plotTitle = title + " Clustering on COVID Numeric Data"
    plt.title(plotTitle)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.colorbar(label='Cluster')
    plt.grid(True)
    plt.show()

def metricsCluster(title):
    print("Metrics for", title, "calculated." )

# Read the numeric dataset
numeric_data = pd.read_csv("COVID_numerics.csv")
print(numeric_data.shape)

# Read the image dataset
image_data = np.loadtxt("COVID_IMG.csv", delimiter=',')

# Reshape each row into a 21x21 matrix if needed
image_matrices = image_data.reshape(-1, 21, 21)
print(image_matrices.shape)  # Should be (number_of_samples, 21, 21)

# Access the first binary image/matrix
first_image = image_matrices[5]  # Shape is (21, 21)

# Flatten the matrix into a 1D array
flattened = first_image.flatten()  # Shape becomes (441,)

"""
# Plot the binary array as a heatmap
plt.figure(figsize=(6, 6))
plt.imshow(first_image, cmap='binary', origin='lower')
plt.title("image")
plt.xlabel("ecg (k)")
plt.ylabel("ecg (k-delay)")
plt.grid(False)
plt.show()
"""

########## CLUSTERING METHODS ##########
# Exclude the target column (assumed to be the last column)
data_for_clustering = numeric_data.iloc[:, :-1]

##### KMEANS #####
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(data_for_clustering)
numeric_data['Cluster'] = kmeans.labels_ # Add cluster labels to the dataset
plotCluster("KMeans")
metricsCluster("KMeans")


##### AGNES #####
agg_clustering = AgglomerativeClustering(n_clusters=3)
numeric_data['Cluster'] = agg_clustering.fit_predict(data_for_clustering)
plotCluster("Agglomerative Clustering")
metricsCluster("Agglomerative Clustering")

##### KMEDOIDS #####
#todo??

##### SUBTRACTIVE CLUSTERING #####
cluster_centers = subtractive_clustering(data_for_clustering)
data_for_clustering = assign_points_to_clusters(data_for_clustering, cluster_centers)

##### DBSCAN #####
dbscan = DBSCAN(eps=0.5, min_samples=5)
cluster_labels = dbscan.fit_predict(data_for_clustering)
numeric_data['Cluster'] = cluster_labels
plotCluster("DBSCAN")
metricsCluster("DBSCAN")


