from pyspark import SparkContext
from pyspark.sql import SparkSession, SQLContext
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from pyspark.ml.feature import VectorAssembler
import sys
import os

# Global Spark variables
spark_context = SparkContext(appName="FrequentItems")
spark_session = SparkSession.builder.appName("Example").getOrCreate()
sql_context = SQLContext(spark_session)

def fetch_songs(input_file, subset='small', percentage=100):
    """Load a specified percentage of the 'conditions.csv' file into a Spark RDD."""
    full_rdd = spark_session.read.option("header", "true").csv(input_file).rdd

    # Convert RDD to DataFrame
    full_df = full_rdd.toDF()

    # Filter the DataFrame to get the specified subset
    filtered_df = full_df.filter(full_df['set32'] == subset)
    sample_df = filtered_df.sample(False, percentage / 100.0, seed=42)
    songs_rdd = sample_df.rdd

    print(f"\n\n Loaded data for subset '{subset}' ({sample_df.count()} out of {full_df.count()} songs)")
    print(songs_rdd.take(5))
    return songs_rdd

def fetch_features(input_file, songs_rdd):
    """Load the features from the 'features.csv' file into a Spark RDD."""
    features_rdd = spark_session.read.option("header", "true").csv(input_file).rdd
    
    # Filter the features RDD based on the track_id in songs_rdd
    features_rdd = features_rdd.filter(lambda x: x['track_id'] in songs_rdd.map(lambda x: x['track_id']).collect())

    print(f"\n\n Loaded features ({features_rdd.count()} songs)")
    return features_rdd

def run_bfr(songs_rdd, features_rdd):
    """Run the BFR algorithm on the given RDD."""
    # Convert RDD to DataFrame
    song_features = features_rdd.toDF()

    # Select the features column
    feature_identifiers = ['tonnetz_skew_06', 'chroma_cqt_kurtosis_09', 'mfcc_max_08', 'spectral_contrast_median_07']
    assembler = VectorAssembler(inputCols=feature_identifiers, outputCol="features_vector")
    song_features = assembler.transform(song_features)

    # Apply agglomerative clustering
    for k in range(8, 17):
        agglomerative = AgglomerativeClustering(k=k, affinity="euclidean")
        model = agglomerative.fit(song_features)

        # Get the cluster labels
        cluster_labels = model.transform(song_features).select("prediction").rdd.flatMap(lambda x: x).collect()

        # Get the cluster centers
        cluster_centers = model.clusterCenters()

        # Calculate radius, diameter, and density
        radius = []
        diameter = []
        density = []
        for i in range(k):
            cluster_points = [song_features[i] for i, label in enumerate(cluster_labels) if label == i]
            cluster_features = [point.features for point in cluster_points]
            cluster_radius = max([point.norm(2) for point in cluster_features])
            cluster_diameter = 2 * cluster_radius
            cluster_density = len(cluster_points) / cluster_diameter

            radius.append(cluster_radius)
            diameter.append(cluster_diameter)
            density.append(cluster_density)

        print(f"Results for k={k}:")
        print("Radius:", radius)
        print("Diameter:", diameter)
        print("Density:", density)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python script.py <input_file> <output_directory>")
        print("Where:")
        print("  <input_file>: The path to the 'conditions.csv.gz' file.")
        print("  <output_directory>: The path to the output directory where the results will be saved.")
        exit(1)

    songs_rdd = fetch_songs(sys.argv[1], "small", percentage=100) 

    features_rdd = fetch_features(sys.argv[1], songs_rdd)

    run_bfr(songs_rdd)

    # Stop the Spark session
    spark_session.stop()