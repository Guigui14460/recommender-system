from recommender_system.data_processor import ProcessData
from recommender_system.recommenders.combination_recommender import CombinationRecommender


data = ProcessData.load_model()
recommender = CombinationRecommender(
    data, "processed_movies_similarity.h5")
print(recommender.recommend(7)[['title', 'movie_id', 'movie']])
