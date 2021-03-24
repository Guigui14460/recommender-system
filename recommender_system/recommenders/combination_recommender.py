from typing import Union
import numpy as np
import pandas as pd

from recommender_system.data_processor.process_data import ProcessData
from .recommender import Recommender
from .content_based_recommender import ContentBasedRecommender
from .collaborative_filtering_recommender import CollaborativeFilteringRecommender


class CombinationRecommender(Recommender):
    """Class used to recommend movies based on other users ratings and content/metadata."""

    def __init__(self, data: ProcessData, content_based_filename: str,
                 collaborative_filename: str, analyze: bool = False) -> None:
        """Initialize the instance.

        Parameters:
        -----------
            data : ProcessDate
                data which can use
            content_based_filename : str
                filename of the file to save or load the cosine similarity
            collaborative_filename : str
                filename of the file to save or load the svd model
            analyze : bool = False
                boolean to know if we need to analyze the data or not
        """
        super().__init__()
        self.data = data
        self.content_based_recommender = ContentBasedRecommender(
            data, content_based_filename, analyze=analyze)
        self.collaborative_filtering_recommender = CollaborativeFilteringRecommender(
            data, collaborative_filename, analyze=analyze)
        print("Combined initialized")

    def recommend(self, user_id: int, nrows: int = None, **kwargs) -> Union[pd.DataFrame, pd.Series]:
        """Recommends movies.

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
        ratings = self.data.get_ratings_by_user_id(user_id)
        print(ratings[['title', 'rating']])
        df_content_based_recommender = self.content_based_recommender.recommend(
            user_id, ratings=ratings)
        df_collaborative_filtering = self.collaborative_filtering_recommender.recommend(
            user_id, ratings=ratings)
        df_content_based_recommender.sort_values("movie_id", inplace=True)
        df_collaborative_filtering.sort_values("movie_id", inplace=True)

        res = df_content_based_recommender
        res['estimations'] += df_collaborative_filtering.estimations
        res = res[~res.movie_id.isin(ratings.movie_id)]
        res.sort_values("estimations", inplace=True, ascending=False)
        return res if nrows is None else res.head(nrows)
