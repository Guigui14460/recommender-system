import pandas as pd

from recommender_system.data_processor import ProcessData
from .recommender import Recommender


class CategoricalRecommender(Recommender):
    """Recommend in function of movies genres."""

    def __init__(self, data: ProcessData) -> None:
        """Initialize the instance.

        Parameters :
        ------------
            data : ProcessData
                the data which we used to recommend
        """
        super().__init__()
        self.data = self.__compute_genres(data.movies)

    def recommend(self, genre: str, percentile: float = .9, nrows: int = None, **kwargs) -> pd.DataFrame:
        """Recommends a list of movies.

        Parameters :
        ------------
            genre : str
                the genre to want to recommend
            percentile : float = 0.9
                mathematics statistics to save only the N percent of vote counts (like median = 50%)
            nrows : int = None
                number of rows to return
        """
        if percentile < 0:
            percentile = abs(percentile)
        if percentile > 1:
            percentile /= percentile
        df = self.data[self.data['genre'] == genre]
        vote_counts = df['vote_count'].astype('int')
        vote_averages = df['vote_average'].astype('int')
        C = vote_averages.mean()
        m = vote_counts.quantile(percentile)

        qualified = df[df['vote_count'] >= m][[
            'title', 'year', 'vote_count', 'vote_average', 'popularity']]
        qualified['vote_count'] = qualified['vote_count'].astype('int')
        qualified['vote_average'] = qualified['vote_average'].astype('int')

        qualified['weighted_rating'] = qualified.apply(lambda x: (
            x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C), axis=1)
        qualified = qualified.sort_values(
            'weighted_rating', ascending=False)

        return qualified if nrows is None else qualified.head(nrows)

    def __compute_genres(self, original_data: pd.DataFrame) -> pd.DataFrame:
        """Computes genres.
        Duplicated row for each genres which are in the row.

        Parameters :
        ------------
            original_data : pd.DataFrame
                the original dataframe

        Returns :
        ---------
            computed_data : pd.DataFrame
                the computed dataframe
        """
        data = original_data.apply(lambda x: pd.Series(
            x['genres'], dtype='object'), axis=1).stack().reset_index(level=1, drop=True)
        data.name = 'genre'
        return original_data.drop('genres', axis=1).join(data)
