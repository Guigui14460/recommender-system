from recommender_system.data_processor import ProcessData
from recommender_system.recommenders import CombinationRecommender


data = ProcessData.load_model()
recommender = CombinationRecommender(
    data, "processed_movies_similarity.h5", "trained_svd_model.pickle", analyze=True)
print("All useful models : SAVED !")
