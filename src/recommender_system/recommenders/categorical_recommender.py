import pandas as pd

from recommender_system.data_processor.process_data import ProcessData
from .recommender import Recommender


class CategoricalRecommender(Recommender):
    def __init__(self, data: ProcessData) -> None:
        super().__init__()
        self.data = self.__compute_genres(data.movies)
        print(self.data.shape)

    def recommend(self, genre: str, percentile: float = .9, first_nrows: int = 250, **kwargs) -> pd.DataFrame:
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
            'weighted_rating', ascending=False).head(first_nrows)

        return qualified

    def __compute_genres(self, original_data: pd.DataFrame) -> pd.DataFrame:
        data = original_data.apply(lambda x: pd.Series(
            x['genres'], dtype='object'), axis=1).stack().reset_index(level=1, drop=True)
        data.name = 'genre'
        return original_data.drop('genres', axis=1).join(data)
