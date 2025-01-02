import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN

import scipy.cluster.hierarchy as shc

def read_data(file_name):
    df = pd.read_csv(file_name, header=None)
    D = df.values
    return df


def normalize_data(data):
    # Step 1: normalize data in [0,1] interval
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    data_norm = (data - data_min) / (data_max - data_min)
    return data_norm

def agglomerative_cluster(file_name, NK):
    data = read_data(file_name)
    clusterH = AgglomerativeClustering(n_clusters=NK)
    clusterH.fit_predict(data)
    labelsH = clusterH.fit_predict(data)
    agglomerative_cluster_plot(data, labelsH)

def dbscan(file_name):
    data = read_data(file_name)
    clusterD = DBSCAN(eps=4, min_samples=5)
    clusterD.fit_predict(data)
    labelsD = clusterD.labels_
    dbscan_cluster_plot(data, labelsD)



# Define subtractive clustering function
def subtractive_clustering(data, radius=2, quash_factor=0.5, min_potential_threshold=0.5):
    # Step 1: Calculate the potential for each point
    potentials = []
    N = data.shape[0]

    radius = 0
    for i in range(N):
        for j in range(N):
            radius += np.linalg.norm(data.values[i] - data.values[j])
    radius /= N ** 2
    alfa = 4 / (radius ** 2)

    for i, point_i in data.iterrows():
        potential = 0
        for j, point_j in data.iterrows():
            distance = np.linalg.norm(point_i - point_j)
            potential += np.exp(- alfa * (distance ** 2))
        potentials.append(potential)
        print(i)

    # Store potentials as a new column in the dataframe
    data['potential'] = potentials

    # __________________________________________________________________-

    # Step 2: Iterate to select cluster centers based on highest potential
    cluster_centers = []
    while True:
        # Select the point with the highest potential
        max_potential_idx = data['potential'].idxmax()
        max_potential = data.loc[max_potential_idx, 'potential']

        # break cycle if densities of potential centers are sufficiently small
        if max_potential < min_potential_threshold:
            break

        # Add the selected point to the list of cluster centers
        # cluster_center = data.loc[max_potential_idx, :-1].values
        cluster_center = data.drop(columns='potential').loc[max_potential_idx].values
        cluster_centers.append(cluster_center)

        # Step 3: Reduce the potential of neighboring points
        for i, point in data.iterrows():
            distance = np.linalg.norm(cluster_center - point[:-1])
            beta = quash_factor * alfa
            reduction_factor = np.exp(-beta * (distance ** 2))
            data.loc[i, 'potential'] -= max_potential * reduction_factor

    return np.array(cluster_centers)


def assign_points_to_clusters(data, cluster_centers):
    # Create an array to store the cluster assignment for each point
    cluster_assignments = []

    # Iterate over each point in the normalized data
    for i, point in data.drop(columns='potential').iterrows():
        print(i)
        # Calculate Euclidean distance from the point to each cluster center
        distances = [np.linalg.norm(point - center) for center in cluster_centers]

        # Assign the point to the cluster with the minimum distance
        closest_cluster_idx = np.argmin(distances)
        cluster_assignments.append(closest_cluster_idx)

    # Return the list of cluster assignments as a new column in the original data
    data['cluster'] = cluster_assignments
    return data

def plot_clusters(data, cluster_centers):
    plt.figure(figsize=(8, 6))

    # Use the cluster column as color labels directly
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=data['cluster'], cmap='viridis', alpha=0.6)

    # Plot cluster centers as black "X" markers
    cluster_centers = np.array(cluster_centers)
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], color='black', marker='x', s=100, label='Centers')

    plt.title('Clusters and their Centers')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True)
    plt.show()

import matplotlib.pyplot as plt
import numpy as np

def agglomerative_cluster_plot(data, labelsH):
    plt.figure(figsize=(8, 6))

    # Scatter plot of the data points, colored by their cluster label
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=labelsH, cmap='viridis', alpha=0.6)

    # Calculate and plot cluster centers
    cluster_centers = data.groupby(labelsH).mean().values
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], color='black', marker='x', s=100, label='Centers')

    plt.title('Agglomerative Clustering Results')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True)
    plt.legend()
    plt.show()

    def dbscan_cluster_plot(data, labelsD):
        plt.figure(figsize=(8, 6))

        # Create a scatter plot of the data points, color-coded by their cluster labels
        plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=labelsD, cmap='viridis', alpha=0.6)

        # Highlight core samples (clusters) with a different marker
        unique_labels = set(labelsD)
        for label in unique_labels:
            if label != -1:  # -1 is the label for noise
                cluster_data = data[labelsD == label]
                plt.scatter(cluster_data.iloc[:, 0], cluster_data.iloc[:, 1], label=f'Cluster {label}')

        # Highlight noise points
        noise_data = data[labelsD == -1]
        plt.scatter(noise_data.iloc[:, 0], noise_data.iloc[:, 1], color='red', marker='x', s=100, label='Noise')

        plt.title('DBSCAN Clustering Results')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.grid(True)
        plt.legend()


def dbscan_cluster_plot(data, labels):
    print('hi')
    #plt.subplot(3,3,2)
    plt.scatter(data.iloc[:,0], data.iloc[:,1], c=labels, cmap='rainbow', s=50)

    plt.title('DBSCAN clustering')
    plt.show()



# run
#data = read_data("P2_CLUSTER2.csv")
#data = normalize_data(data)
#cluster_centers = subtractive_clustering(data)
#data = assign_points_to_clusters(data, cluster_centers)
#plot_clusters(data, cluster_centers)

agglomerative_cluster("P2_CLUSTER4.csv", 2)
dbscan("P2_CLUSTER4.csv")