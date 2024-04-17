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

def fetch_data(input_file, percentage=5):
    """Load a specified percentage of the 'conditions.csv.gz' file into a Spark RDD."""
    full_rdd = spark_session.read.option("header", "true").csv(input_file).rdd
    total_count = full_rdd.count()
    sample_rdd = full_rdd.sample(False, percentage / 100.0, seed=42)
    conditions_rdd = sample_rdd
    print(f"\n\n Loaded {percentage}% of the data ({sample_rdd.count()} out of {total_count} rows)")
    print(conditions_rdd.take(5))
    return conditions_rdd


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_file> <output_directory>")
        print("Where:")
        print("  <input_file>: The path to the 'conditions.csv.gz' file.")
        print("  <output_directory>: The path to the output directory where the results will be saved.")
        exit(1)

    conditions_rdd = fetch_data(sys.argv[1], percentage=6) 

    # Stop the Spark session
    spark_session.stop()