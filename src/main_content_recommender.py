from recommender_system.data_processor import ProcessData
from recommender_system.recommenders.content_based_recommender import ContentBasedRecommender


data = ProcessData.load_model()
recommender = ContentBasedRecommender(
    data, "processed_movies_similarity.h5", analyze=True)
print(recommender.recommend("Star Wars", nrows=20))
print(recommender.recommend("Toy Story", nrows=20))
