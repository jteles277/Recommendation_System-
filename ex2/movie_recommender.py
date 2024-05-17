import sys
import os 
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, split, corr
from pyspark.ml.evaluation import RegressionEvaluator
from utils import calculate_percentage_error

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python script.py <movies_file> <ratings_file> <output_directory>")
        exit(1)

    spark = SparkSession.builder.appName("MovieRecommender").getOrCreate()

    # Create results directory if it does not exist
    folder_path = "results"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Load movies data
    movies_df = spark.read.csv(sys.argv[1], header=True, inferSchema=True)
    movies_df = movies_df.select(col("*"), explode(split(col("genres"), "|")).alias("genre")) \
                         .drop("genres") \
                         .withColumn("movieId", col("movieId").cast("string")) \
                         .withColumn("title", col("title").cast("string")) \
                         .toDF("movieId", "title", "genre")

    # Read ratings data
    ratings_df = spark.read.csv(sys.argv[2], header=True, inferSchema=True)
    
    # Randomly select and hide 10% of ratings for validation
    hidden_ratings = ratings_df.sample(fraction=0.1, seed=42)
    print(f"\n\nHidden Ratings: ({hidden_ratings.count()})")
    visible_ratings = ratings_df.subtract(hidden_ratings)
    print(f"\n\nVisible Ratings: ({visible_ratings.count()})")

    start_time = time.time()  # Record start time

    # Join ratings and movies data
    joined_data = ratings_df.join(movies_df, on="movieId", how="inner")

    # Group ratings by movie and calculate the average rating for each movie
    average_ratings = joined_data.groupby("movieId").avg("rating")

    # Convert the joined DataFrame to an RDD
    joined_rdd = joined_data.rdd

    # Pritn the average ratings of the first 10 movies
    # print("\n\nAverage Ratings:")
    # average_ratings.show(10)
    # print("\n\n")

    # Normalize ratings by subtracting the average rating for each movie
    normalized_ratings = ratings_df.join(average_ratings, on="movieId") \
            .withColumnRenamed("avg(rating)", "avgRating")

    # Print the first 10 rows of the joined DataFrame
    # print("\n\nNormalized Ratings:")
    # normalized_ratings.show(10)
    # print("\n\n")

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

    # Broadcast the ratings_df and movie_pairs DataFrames
    # broadcast_ratings_df = spark.sparkContext.broadcast(ratings_df)
    # broadcast_movie_pairs = spark.sparkContext.broadcast(movie_pairs)

    def predict_rating(user_id, movie_id):

        # ratings_df_broadcasted = broadcast_ratings_df.value
        # movie_pairs_broadcasted = broadcast_movie_pairs.value

        # Find movies rated by the given user
        user_movies = ratings_df.where(col("userId") == user_id).select("movieId").rdd.flatMap(lambda x: x).collect()
        
        # Find movies similar to the given movie that were also rated by the user
        similarities = movie_pairs \
            .where((col("movieId1").isin(user_movies)) & (col("movieId2") == movie_id) & (col("similarity") > 0) & (col("similarity").isNotNull())) \
            .select("movieId1", "movieId2", "similarity") \
            .collect()

         # Print the first 10 movie pairs
        print(f"Found {len(similarities)} similar movies for movie {movie_id}:")
        print(f"\n\nSimilarities ({len(similarities)}):")
        for row in similarities:
            print(f"Movie {row['movieId1']} - Movie {row['movieId2']}: {row['similarity']}")

        print("\n\n")
        if not similarities:
            return 0

        # Calculate the predicted rating
        numerator = 0
        denominator = 0

        for neighbor in similarities:
            sim_movie_id = neighbor["movieId1"]
            #print(f"Similarity: {neighbor['similarity']}")
            sim = neighbor["similarity"]
            #print(f"Similar Movie ID: {sim_movie_id}")
            #print(f"User ID: {user_id}")
            #print(f"Movie ID: {sim_movie_id}")
            rating = ratings_df.where((col("userId") == user_id) & (col("movieId") == sim_movie_id)).select("rating").first()
            #print(f"Neighbor Rating: {rating}")
            if rating is not None:
                numerator += sim * rating[0]
                denominator += abs(sim)

        predicted_rating = numerator / denominator if denominator != 0 else 0
        #print(predict_rating)
        #sys.exit(0)
        return predicted_rating

    # Make predictions and evaluate model
    predictions = []
    for row in hidden_ratings.collect():
        user_id = row["userId"]
        movie_id = row["movieId"]
        real_rating = row["rating"]
        predicted_rating = predict_rating(user_id, movie_id)
        rmse = calculate_percentage_error(real_rating, predicted_rating)
        predictions.append((user_id, movie_id, predicted_rating))

        # Write to a file in the output directory
        with open(sys.argv[3], "a") as f:
            f.write(f"Predicted rating for user {user_id} and movie {movie_id}: {predicted_rating:.2f} | real rating = {real_rating:.1f} | RMSE = {rmse:.2f}\n")

        print(f"\nPredicted rating for user {user_id} and movie {movie_id}: {predicted_rating} | real rating = {real_rating} | RMSE = {rmse}\n")

    end_time = time.time()  # Record end time
    execution_time = end_time - start_time  
    times_results_file = "results/execution_times.txt"
    with open(times_results_file, "a") as f:
        f.write(f"Execution time: {execution_time:.2f} seconds for file {sys.argv[1]}\n")
    
    #predictions_df = spark.createDataFrame(predictions, ["userId", "movieId", "prediction"])

    # Evaluate the predictions using RMSE
    #evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    #rmse = evaluator.evaluate(predictions_df)
    #print("Root Mean Squared Error (RMSE) for hidden ratings = " + str(rmse))

