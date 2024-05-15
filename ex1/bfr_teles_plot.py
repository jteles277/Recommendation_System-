from pyspark import SparkContext
from pyspark.sql import SparkSession, SQLContext
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import StandardScaler
import sys
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 

# Global Spark variables
spark_context = SparkContext(appName="BFR")
spark_session = SparkSession.builder.appName("Example").getOrCreate()
sql_context = SQLContext(spark_session)

def fetch_data(tracks_file, features_file, subset='small'):
    """Load the features from the 'features.csv' file into a Spark RDD."""
    songs_df = pd.read_csv(tracks_file, header=[0,1], index_col=0)

    small_index = songs_df[('set', 'subset')] == 'small'

    small_df = pd.read_csv(features_file, header=[0,1,2], index_col=0).loc[small_index]

    drop_columns = [96,97,98,99,100,101,102,103,104,105,106,107,180,181,182,183,184,185,186,187,188,189,190,191]

    # Assign the result of drop() back to small_df
    small_df = small_df.drop(columns=small_df.columns[drop_columns])

    print(f"\n\n Loaded data for subset '{subset}'")
    small_df.head()

    return small_df

def find_best_k(features_df):
    """Run Agglomerative Clustering to find the best k."""
    # Scale the features
    scaler = StandardScaler()
    song_features = scaler.fit_transform(features_df)

    # Apply agglomerative clustering
    silhouette_scores = []
    k_values = list(range(8, 17))
    for k_value in k_values:
        agglomerative = AgglomerativeClustering(n_clusters=k_value, metric="euclidean")
        model = agglomerative.fit(song_features)
        silhouette_scores.append(silhouette_score(song_features, agglomerative.labels_))

        # Get cluster labels 
        cluster_labels = model.labels_

        # Calculate radius, diameter, and density
        radius = []
        diameter = []
        density = []
        for i in range(model.n_clusters):
            centroid = np.mean(song_features[cluster_labels == i], axis=0)
            cluster_points = song_features[cluster_labels == i]
            cluster_radius = max([np.linalg.norm(point - centroid) for point in cluster_points])
            # cluster_distances = pairwise_distances(X, metric='euclidean')
            cluster_density = len(cluster_points) / (np.pi * cluster_radius**2)
            radius.append(cluster_radius)
            # diameter.append(cluster_diameter)
            density.append(cluster_density)

        average_density = np.mean(density)
        if average_density > cluster_density:
            cluster_density = average_density
            best_k = k_value 

    return best_k

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

 
class BFRCluster:
    def __init__(self, k, m):
        self.k = k
        self.m = m
        self.centroids = None
        self.clusters = None

    def fit(self, data_file, iterations):
        self.load_data(data_file)
        self.initialize_centroids()
        for _ in range(iterations):
            self.assign_clusters()
            self.update_centroids()
            self.reassign_clusters()

    def load_data(self, file_path):
        # Load CSV file into pandas DataFrame
        self.data = pd.read_csv(file_path, header=[0,1,2,3], chunksize=100)
        for chunk in self.data:
            a = chunk
        self.data = a
        self.num_features = len(self.data.columns)
        print("data loaded")

    def initialize_centroids(self):
        self.centroids = self.data.sample(n=self.k, random_state=42)

    def assign_clusters(self):
        distances = np.zeros((len(self.data), self.k))
        for i in range(self.k):
            distances[:, i] = np.linalg.norm(self.data - self.centroids.iloc[i], axis=1)
        self.clusters = np.argmin(distances, axis=1)

    def update_centroids(self):
        for i in range(self.k):
            self.centroids.iloc[i] = self.data[self.clusters == i].mean()

    def reassign_clusters(self):
        distances = np.zeros((len(self.data), self.k))
        for i in range(self.k):
            distances[:, i] = np.linalg.norm(self.data - self.centroids.iloc[i], axis=1)
        new_clusters = np.argmin(distances, axis=1)
        changed = sum(new_clusters != self.clusters)
        if changed < self.m:
            self.clusters = new_clusters

    def get_clusters(self):
        clusters = []
        for i in range(self.k):
            cluster_data = self.data[self.clusters == i]
            clusters.append(cluster_data)
        return clusters

    def plot_clusters(self):
        centroids = [cluster.mean() for cluster in self.clusters]
        centroids_x = [centroid[0] for centroid in centroids]
        centroids_y = [centroid[1] for centroid in centroids]

        # Scatter plot of data points
        plt.scatter(self.data.iloc[:, 0], self.data.iloc[:, 1], alpha=0.5, label='Data Points')
        
        # Plot centroids
        plt.scatter(centroids_x, centroids_y, color='red', marker='o', label='Centroids')
        
        plt.title('Cluster Visualization')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.show()
 
    def extract_genres(self, track_ids):
        
        print(tracks_df.columns)
        genre_rows = []
        for track_id in track_ids:
            genre_row = tracks_df.loc[tracks_df['track'] == track_id, ('track',          'genre_top')]
            genre_rows.append(genre_row)
        return genre_rows


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python script.py <input_file> <output_directory>")
        print("Where:")
        print("  <input_file>: The path to the 'conditions.csv.gz' file.")
        print("  <output_directory>: The path to the output directory where the results will be saved.")
        exit(1)
    
    tracks_file = sys.argv[1]    
    features_file = sys.argv[2]
    
    # Load songs metadata
    tracks_df = pd.read_csv(tracks_file, header=[0,1,2])
    
    print(tracks_df.head()) 
 
    #features_df = fetch_data(sys.argv[1], features_file, "small")  
    
    #best_k = find_best_k(features_df)
    #print(f"\n\n The best k is: {best_k}")

    # Example usage:
    bfr = BFRCluster(k=16, m=10)  # Number of clusters and memory size
    bfr.fit(features_file, iterations=10)
    clusters = bfr.get_clusters()
    
    for i, cluster in enumerate(clusters):
        print(f'Cluster {i+1}:')
        print(cluster)
        print(cluster.columns)
        print('\n')
    
    bfr.plot_clusters()
    
    
    # Extract track IDs for each cluster
    cluster_track_ids = []
    for cluster in bfr.get_clusters():
        # Access the 'track_id' column by specifying both levels of the multi-level index
        track_ids = cluster[('feature', 'statistics', 'number', 'track_id')].tolist()  # Assuming 'track_id' is the correct column name
        cluster_track_ids.append(track_ids)

    # Print cluster track IDs
    for i, track_ids in enumerate(cluster_track_ids):
        print(f'Cluster {i+1} Track IDs:', track_ids)
        
    
    # Extract genres for each cluster
    cluster_genres = []
    for cluster in bfr.get_clusters():
        track_ids = cluster[('feature', 'statistics', 'number', 'track_id')].tolist()  # Adjust column name if needed
        cluster_genres.append(bfr.extract_genres(track_ids))

    # Count occurrences of each genre in each cluster
    genre_counts = [{genre: cluster_genres[i].count(genre) for genre in set(cluster_genres[i])} for i in range(len(cluster_genres))]

    # Plot most common genres in each cluster
    for i, counts in enumerate(genre_counts):
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]  # Top 5 genres
        genres, counts = zip(*sorted_counts)
        plt.figure(figsize=(8, 6))
        plt.barh(genres, counts)
        plt.xlabel('Count')
        plt.ylabel('Genre')
        plt.title(f'Most Common Genres in Cluster {i+1}')
        plt.gca().invert_yaxis()  # Invert y-axis to have the highest count at the top
        plt.show()