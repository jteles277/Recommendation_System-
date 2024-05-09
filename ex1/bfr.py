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


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python script.py <input_file> <output_directory>")
        print("Where:")
        print("  <input_file>: The path to the 'conditions.csv.gz' file.")
        print("  <output_directory>: The path to the output directory where the results will be saved.")
        exit(1)

    features_df = fetch_data(sys.argv[1], sys.argv[2], "small") 

    best_k = find_best_k(features_df)
    print(f"\n\n The best k is: {best_k}")

    # Stop the Spark session
    spark_session.stop()