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
from collections import Counter

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


class BFRCluster:
    
    def __init__(self, k, distance_threshold):
        self.k = k 
        self.distance_threshold = distance_threshold
        self.summaries = None
        self.outliers = []
        self.cluster_assignments = None
        self.track_ids = []

    def fit(self, data_file, chunk_size=1000, iterations=10):
        last_chunk = None
        last_track_ids_chunk = None
        i = 0 
        for chunk in pd.read_csv(data_file, header=[0,1,2,3], chunksize=chunk_size):
            
            i += 1
             
            info = ""
            if self.summaries is not None:
                for cluster in self.summaries:
                    info += " | " + str(cluster['N']) 
            
            print("\n", chunk_size*i, " Points loaded -> ", info)        
            
            chunk, track_ids_chunk = self.clean_data(chunk)
            self.track_ids.extend(track_ids_chunk)  # Extend track_ids list
            if self.summaries is None:
                self.initialize_centroids(chunk)
            else:
                self.process_chunk(chunk, track_ids_chunk, iterations)
            last_chunk = chunk
            last_track_ids_chunk = track_ids_chunk

        # Ensure the last chunk is processed
        if last_chunk is not None:
            self.process_chunk(last_chunk, last_track_ids_chunk, iterations)

        # Perform final merge to ensure number of clusters is exactly self.k
        self.merge_clusters_to_k()
        
        print("BFR clustering completed")
        
    def clean_data(self, chunk):  
        track_ids_chunk = chunk['feature', 'statistics', 'number', 'track_id'].values
        #chunk = chunk.drop(columns=['feature', 'statistics', 'number', 'track_id'], errors='ignore')
        chunk = chunk.apply(pd.to_numeric, errors='coerce').dropna()
        print(f"Data cleaned: {chunk.shape} rows, {chunk.shape[1]} columns")
        return chunk, track_ids_chunk

    def initialize_centroids(self, data_chunk):
        centroids = data_chunk.sample(n=self.k, random_state=42).to_numpy()
        self.summaries = []
        for centroid in centroids:
            self.summaries.append({
                'N': 1,
                'SUM': centroid,
                'SUMSQ': centroid**2,
                'track_ids'  : []
            })
        print(f"Initialized centroids: {len(self.summaries)} centroids")

    def process_chunk(self, data_chunk, track_ids_chunk, iterations):
        data_np = data_chunk.to_numpy()
        print(f"Processing chunk: {data_np.shape} rows, {data_np.shape[1]} columns")
        for _ in range(iterations):
            self.assign_clusters(data_np, track_ids_chunk)
            self.update_summaries(data_np, track_ids_chunk)
            self.reassign_outliers(data_np)
            self.merge_clusters_to_k()

    def assign_clusters(self, data, track_ids_chunk):
        distances = np.zeros((len(data), self.k))

        for i, summary in enumerate(self.summaries):
            centroid = summary['SUM'] / summary['N']
            distances[:, i] = np.linalg.norm(data - centroid, axis=1)
        
        self.clusters = np.argmin(distances, axis=1)
        self.outliers = [i for i, dists in enumerate(distances) if np.min(dists) > self.distance_threshold]
        self.track_ids_chunk = track_ids_chunk

    def update_summaries(self, data, track_ids_chunk):
        for i in range(self.k):
            points_in_cluster = data[self.clusters == i]
            track_ids_in_cluster = [track_ids_chunk[j] for j in range(len(track_ids_chunk)) if self.clusters[j] == i]
            if len(points_in_cluster) == 0:
                continue
            N = len(points_in_cluster)
            SUM = points_in_cluster.sum(axis=0)
            SUMSQ = (points_in_cluster**2).sum(axis=0)
            self.summaries[i]['N'] += N
            self.summaries[i]['SUM'] += SUM
            self.summaries[i]['SUMSQ'] += SUMSQ
            if 'track_ids' in self.summaries[i]:
                self.summaries[i]['track_ids'].extend(track_ids_in_cluster)
            else:
                self.summaries[i]['track_ids'] = track_ids_in_cluster 

    def reassign_outliers(self, data):
        outliers_data = data[self.outliers]
        if len(outliers_data) > 0:
            new_centroids = self.create_new_clusters(outliers_data)
            self.summaries.extend(new_centroids)
        self.outliers = []

    def create_new_clusters(self, data):
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=min(self.k, len(data)), n_init=10)
        kmeans.fit(data)
        new_summaries = []
        for i in range(min(self.k, len(data))):
            points_in_cluster = data[kmeans.labels_ == i]
            N = len(points_in_cluster)
            SUM = points_in_cluster.sum(axis=0)
            SUMSQ = (points_in_cluster**2).sum(axis=0)
            new_summaries.append({
                'N': N,
                'SUM': SUM,
                'SUMSQ': SUMSQ,
                'track_ids'  : []
            })
        #print(f"Created new clusters: {len(new_summaries)} clusters")
        return new_summaries

    def merge_clusters_to_k(self):
        while len(self.summaries) > self.k:
            distances = np.full((len(self.summaries), len(self.summaries)), np.inf)
            for i in range(len(self.summaries)):
                for j in range(i + 1, len(self.summaries)):
                    centroid_i = self.summaries[i]['SUM'] / self.summaries[i]['N']
                    centroid_j = self.summaries[j]['SUM'] / self.summaries[j]['N']
                    distance = np.linalg.norm(centroid_i - centroid_j)
                    distances[i, j] = distance
                    distances[j, i] = distance

            min_dist_idx = np.unravel_index(np.argmin(distances, axis=None), distances.shape)
            i, j = min_dist_idx

            self.summaries[i]['N'] += self.summaries[j]['N']
            self.summaries[i]['SUM'] += self.summaries[j]['SUM']
            self.summaries[i]['SUMSQ'] += self.summaries[j]['SUMSQ']
            self.summaries[i]['track_ids'].extend(self.summaries[j]['track_ids'])

            del self.summaries[j]
            #print(f"Clusters after merging: {len(self.summaries)}")
 
 

        
    def get_clusters(self):
        clusters = []
        for i in range(len(self.summaries)):
            centroid = self.summaries[i]['N']
            track_ids = self.summaries[i].get('track_ids', [])
            clusters.append({'centroid': centroid, 'track_ids': track_ids})
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
         
        # Extract genres for the given track IDs
        track_id_to_genre = tracks_df['track']['genre_top']
        track_id_to_genre = track_id_to_genre[small_index.index] 
        tracks_to_keep = track_id_to_genre[track_id_to_genre.index.isin(track_ids)]
         
         
        return tracks_to_keep


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
    tracks_df = pd.read_csv(tracks_file, header=[0,1], index_col=0)
    small_index = tracks_df[('set', 'subset')] == 'small'
 
    # Example usage: 
    bfr = BFRCluster(k=16, distance_threshold=1420.0)
    bfr.fit(features_file, iterations=3, chunk_size=1000)
    clusters = bfr.get_clusters() 
    
    # Extract genres for each cluster
    cluster_genres = []
    for cluster in clusters:
        track_ids = cluster['track_ids']
        cluster_genres.append(bfr.extract_genres(track_ids))
        
    cluster_genres = [[genre for genre in cluster if pd.notna(genre)] for cluster in cluster_genres]
     

    # Function to plot pie chart for a cluster 
    def plot_pie_chart(ax, cluster_genres, cluster_index):
        genre_counts = Counter(cluster_genres)  

        # Group genres with less than 10% occurrence into "Others"
        total_count = sum(genre_counts.values())
        threshold = total_count * 0.1
        filtered_genre_counts = {genre: count for genre, count in genre_counts.items() if count >= threshold}
        others_count = total_count - sum(filtered_genre_counts.values())
        if others_count > 0:
            filtered_genre_counts['Others'] = others_count

        labels = list(filtered_genre_counts.keys())
        sizes = list(filtered_genre_counts.values())
        
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        ax.set_title(f'Cluster {cluster_index} - Genre Distribution')
        ax.axis('equal') 

    # Create subplots
    num_clusters = len(cluster_genres)
    num_cols = 4
    num_rows = (num_clusters + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))

    # Plot pie chart for each cluster
    for i, genres in enumerate(cluster_genres, start=1):
        row_index = (i - 1) // num_cols
        col_index = (i - 1) % num_cols
        if num_rows == 1:
            ax = axes[col_index]
        else:
            ax = axes[row_index, col_index]
        plot_pie_chart(ax, genres, i)
 
    plt.show()




        