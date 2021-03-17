from recommender_system.data_processor.process_data import ProcessData
from recommender_system.recommenders.categorical_recommender import CategoricalRecommender


data = ProcessData.load_model()
print(data.movies.head())
# print(data.movies[data.movies['title'] == 'Titanic'])
# category_rec = CategoricalRecommender(data)
# print(category_rec.recommend("Action", first_nrows=15))
# print(category_rec.recommend("Horror"))
# print(category_rec.recommend("Romance"))
