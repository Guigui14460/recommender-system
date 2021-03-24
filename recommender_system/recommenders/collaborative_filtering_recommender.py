import pickle
from typing import Union

import pandas as pd
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate

from recommender_system.data_processor.process_data import ProcessData
from .recommender import Recommender


class CollaborativeFilteringRecommender(Recommender):
    """Class used to recommend movies based on other users ratings."""

    def __init__(self, data: ProcessData, filename: str, analyze: bool = False) -> None:
        """Initialize the instance.

        Parameters:
        -----------
            data : ProcessDate
                data which can use
            filename : str
                filename of the file to save or load the svd model
            analyze : bool = False
                boolean to know if we need to analyze the data or not
        """
        super().__init__()
        self.data = data
        if analyze:
            self.svd = self.__train_model(filename)
        else:
            self.svd = self.__load_model(filename)
        print("Collaborative filtering initialized")

    def __train_model(self, filename: str) -> SVD:
        """Trains the SVD model with the ratings data.

        Parameters:
        -----------
            filename : str
                filename of the file where to save the svd model

        Returns:
        --------
            svd: SVD
                SVD trained model
        """
        reader = Reader()
        svd = SVD()
        print("Loading dataset ...")
        data = Dataset.load_from_df(
            self.data.ratings[['user_id', 'movie_id', 'rating']], reader)
        print("Cross validation ...")
        cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5)
        trainset = data.build_full_trainset()
        print("Training the SVD ...")
        svd.fit(trainset)
        print("SVD ready !")
        dico = {
            "svd": svd
        }
        pickle.dump(dico, open(filename, "wb"))
        return svd

    def __load_model(self, filename: str) -> SVD:
        """Loads the SVD model from file.

        Parameters:
        -----------
            filename : str
                filename of the file where to read the saved svd model

        Returns:
        --------
            svd: SVD
                SVD trained model
        """
        return pickle.load(open(filename, "rb"))["svd"]

    def recommend(self, user_id: int, nrows: int = None, **kwargs) -> Union[pd.DataFrame, pd.Series]:
        """Recommends movies based on other users.

        Parameters:
        -----------
            user_id : int
                user identifiant for the recommendation
            nrows : int = None
                number of rows to return

        Returns:
        --------
            the recommended movies in a dataframe or a series
        """
        movie_ids = kwargs.get(
            "ratings", self.data.get_ratings_by_user_id(user_id))
        self.data.movies['estimations'] = self.data.movies.movie_id.apply(lambda x: self.svd.predict(
            user_id, x).est)
        sorted_estimations = self.data.movies.sort_values(
            'estimations', ascending=False)
        sorted_estimations = sorted_estimations[~sorted_estimations.movie_id.isin(
            movie_ids.movie_id)]
        return sorted_estimations if nrows is None else sorted_estimations.head(nrows)
