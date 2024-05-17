import sys
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, split, corr
from pyspark.ml.evaluation import RegressionEvaluator

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python script.py <movies_file> <ratings_file> <output_directory>")
        exit(1)

    spark = SparkSession.builder.appName("MovieRecommender").getOrCreate()

    # Load movies data
    movies_df = spark.read.csv(sys.argv[1], header=True, inferSchema=True)
    movies_df = movies_df.select(col("*"), explode(split(col("genres"), "|")).alias("genre")) \
                         .drop("genres") \
                         .withColumn("movieId", col("movieId").cast("string")) \
                         .withColumn("title", col("title").cast("string")) \
                         .toDF("movieId", "title", "genre")

    # Read ratings data
    ratings_df = spark.read.csv(sys.argv[2], header=True, inferSchema=True)
    train_df, test_df = ratings_df.randomSplit([0.9, 0.1], seed=42)

    # Join ratings and movies data
    joined_data = train_df.join(movies_df, on="movieId", how="inner")

    # Print the first 10 rows of the joined DataFrame
    print("\n\nJoined Data:")
    joined_data.show(10)
    print("\n\n")

    # Group ratings by movie and calculate the average rating for each movie
    average_ratings = joined_data.groupby("movieId").avg("rating")

    # Convert the joined DataFrame to an RDD
    joined_rdd = joined_data.rdd

    # Pritn the average ratings of the first 10 movies
    print("\n\nAverage Ratings:")
    average_ratings.show(10)
    print("\n\n")

    # Normalize ratings by subtracting the average rating for each movie
    normalized_ratings = train_df.join(average_ratings, on="movieId") \
            .withColumnRenamed("avg(rating)", "avgRating")

    # Print the first 10 rows of the joined DataFrame
    print("\n\nNormalized Ratings:")
    normalized_ratings.show(10)
    print("\n\n")

    normalized_ratings = normalized_ratings.withColumn("rating_norm", col("rating") - col("avgRating"))

    # Calculate Pearson correlation between each pair of movies
    movie_pairs = normalized_ratings \
        .selectExpr("userId", "movieId as movieId1", "rating_norm as rating_norm1") \
        .join(
            normalized_ratings.selectExpr("userId", "movieId as movieId2", "rating_norm as rating_norm2"),
            on="userId"
        ) \
        .filter(col("movieId1") < col("movieId2")) \
        .groupby("movieId1", "movieId2") \
        .agg(corr("rating_norm1", "rating_norm2").alias("similarity"))

    # Print the first 10 movie pairs
    print("\n\nNormalized Movie Pairs:")
    movie_pairs.show(10)
    print("\n\n")

    def get_user_movie_rating(user_id, movie_id, train_df):
        rating = train_df.where((col("userId") == user_id) & (col("movieId") == movie_id)).select("rating").first()
        print(f"Rating for user {user_id} and movie {movie_id}: {rating[0] if rating else None}")
        return rating[0] if rating else None

    def predict_rating(user_id, movie_id):
        # Find movies similar to the given movie
        similarities = movie_pairs \
            .where((col("movieId1") == movie_id) & (col("similarity") > 0) & (col("similarity").isNotNull())) \
            .select("movieId2", "similarity") \
            .collect()

        
        if not similarities:
            return 0

        # Calculate the predicted rating
        numerator = 0
        denominator = 0

        for row in similarities:
            sim_movie_id = row["movieId2"]
            sim = row["similarity"]
            rating = get_user_movie_rating(user_id, sim_movie_id, train_df)
            if rating is not None:
                numerator += sim * rating
                denominator += abs(sim)

        predicted_rating = numerator / denominator if denominator != 0 else 0

        print("\n\n")
        print(f"Predicted rating for user {user_id} and movie {movie_id}: {predicted_rating}")
        print("\n\n")
        return predicted_rating
    
    # Print some predictions
    print("Predictions:")
    for user_id, movie_id, rating in test_df.select("userId", "movieId", "rating").take(10):
        print(f"User {user_id} - Movie {movie_id}: {predict_rating(user_id, movie_id)}")

    # Make predictions and evaluate model
    predictions = test_df \
        .rdd \
        .map(lambda row: (row["userId"], row["movieId"], predict_rating(row["userId"], row["movieId"]))) \
        .toDF(["userId", "movieId", "prediction"])
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    print("Root Mean Squared Error (RMSE) = " + str(rmse))


import sys
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, split, corr
from pyspark.ml.evaluation import RegressionEvaluator

# Constants
SPLIT_CHAR = "|"
SIMILARITY_THRESHOLD = 0

def load_movies_data(movies_file):
    movies_df = spark.read.csv(movies_file, header=True, inferSchema=True)
    movies_df = movies_df.select(col("*"), explode(split(col("genres"), SPLIT_CHAR)).alias("genre")) \
                         .drop("genres") \
                         .withColumn("movieId", col("movieId").cast("string")) \
                         .withColumn("title", col("title").cast("string")) \
                         .toDF("movieId", "title", "genre")
    return movies_df

def load_ratings_data(ratings_file):
    ratings_df = spark.read.csv(ratings_file, header=True, inferSchema=True)
    train_df, test_df = ratings_df.randomSplit([0.9, 0.1], seed=42)
    return train_df, test_df

def get_average_ratings(joined_data):
    average_ratings = joined_data.groupby("movieId").avg("rating")
    return average_ratings

def normalize_ratings(train_df, average_ratings):
    normalized_ratings = train_df.join(average_ratings, on="movieId") \
            .withColumnRenamed("avg(rating)", "avgRating") \
            .withColumn("rating_norm", col("rating") - col("avgRating"))
    return normalized_ratings

def calculate_movie_similarities(normalized_ratings):
    movie_pairs = normalized_ratings \
        .selectExpr("userId", "movieId as movieId1", "rating_norm as rating_norm1") \
        .join(
            normalized_ratings.selectExpr("userId", "movieId as movieId2", "rating_norm as rating_norm2"),
            on="userId"
        ) \
        .filter(col("movieId1") < col("movieId2")) \
        .groupby("movieId1", "movieId2") \
        .agg(corr("rating_norm1", "rating_norm2").alias("similarity"))
    return movie_pairs

def get_user_movie_rating(user_id, movie_id, train_df):
    rating = train_df.where((col("userId") == user_id) & (col("movieId") == movie_id)).select("rating").first()
    return rating[0] if rating else None

def predict_rating(user_id, movie_id, movie_pairs, train_df):
    similarities = movie_pairs \
        .where((col("movieId1") == movie_id) & (col("similarity") > SIMILARITY_THRESHOLD) & (col("similarity").isNotNull())) \
        .select("movieId2", "similarity") \
        .collect()

    if not similarities:
        return 0

    numerator = 0
    denominator = 0

    for row in similarities:
        sim_movie_id = row["movieId2"]
        sim = row["similarity"]
        rating = get_user_movie_rating(user_id, sim_movie_id, train_df)
        if rating is not None:
            numerator += sim * rating
            denominator += abs(sim)

    predicted_rating = numerator / denominator if denominator != 0 else 0
    return predicted_rating

def evaluate_model(test_df, movie_pairs, train_df):
    predictions = test_df \
        .rdd \
        .map(lambda row: (row["userId"], row["movieId"], predict_rating(row["userId"], row["movieId"], movie_pairs, train_df))) \
        .toDF(["userId", "movieId", "prediction"])
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    return rmse

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python script.py <movies_file> <ratings_file> <output_directory>")
        exit(1)

    spark = SparkSession.builder.appName("MovieRecommender").getOrCreate()

    movies_df = load_movies_data(sys.argv[1])
    train_df, test_df = load_ratings_data(sys.argv[2])

    joined_data = train_df.join(movies_df, on="movieId", how="inner")
    average_ratings = get_average_ratings(joined_data)
    normalized_ratings = normalize_ratings(train_df, average_ratings)
    movie_pairs = calculate_movie_similarities(normalized_ratings)

    rmse = evaluate_model(test_df, movie_pairs, train_df)
    print("Root Mean Squared Error (RMSE) = " + str(rmse))
