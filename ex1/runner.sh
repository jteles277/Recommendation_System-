spark-submit src/BFR.py fma_metadata/tracks.csv fma_metadata/features.csv results 
spark-submit src/find_best_k.py fma_metadata/tracks.csv fma_metadata/features.csv results 

spark-submit src/BFR.py fma_metadata/tracks.csv fma_metadata/features.csv results 