from recommender_system.data_processor import ProcessData
from recommender_system.recommenders.categorical_recommender import CategoricalRecommender
from recommender_system.recommenders.content_based_recommender import ContentBasedRecommender


data = ProcessData.load_model()
print(data.movies.head())
# print(data.movies[data.movies['title'] == 'Titanic'])
# category_rec = CategoricalRecommender(data)
# print(category_rec.recommend("Action", first_nrows=15))
# print(category_rec.recommend("Horror"))
# print(category_rec.recommend("Romance"))

recommender = ContentBasedRecommender(data, "processed_movies_similarity.h5")
print(recommender.recommend("Star Wars", nrows=20))
print(recommender.recommend("Toy Story", nrows=20))
