from recommender_system.data_processor import ProcessData
from recommender_system.recommenders import CombinationRecommender


data = ProcessData.load_model()

recommender = CombinationRecommender(
    data, "processed_movies_similarity.h5", "trained_svd_model.pickle")
print(recommender.recommend(25, nrows=20)[
      ['title', 'movie_id', 'estimations']])

# TODO: select user from 1 to 671
