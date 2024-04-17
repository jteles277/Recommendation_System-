from pyspark import SparkContext
from pyspark.sql import SparkSession, SQLContext
from itertools import combinations
from fractions import Fraction
import sys
import os

# Global Spark variables
spark_context = SparkContext(appName="FrequentItems")
spark_session = SparkSession.builder.appName("Example").getOrCreate()
sql_context = SQLContext(spark_session)

def fetch_data(input_file, subset='small', percentage=100):
    """Load a specified percentage of the 'conditions.csv.gz' file into a Spark RDD."""
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


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_file> <output_directory>")
        print("Where:")
        print("  <input_file>: The path to the 'conditions.csv.gz' file.")
        print("  <output_directory>: The path to the output directory where the results will be saved.")
        exit(1)

    songs_rdd = fetch_data(sys.argv[1], "small", percentage=100) 

    # Stop the Spark session
    spark_session.stop()