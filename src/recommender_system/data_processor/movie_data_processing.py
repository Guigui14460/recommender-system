from ast import literal_eval

import numpy as np
import pandas as pd

from recommender_system import constants


class MoviesDataProcessing:
    def __init__(self, data_directory: str = constants.DATA_DIRECTORY) -> None:
        self.data_directory = data_directory
        self.movies = None

    def __get_file(self, filename: str) -> str:
        return self.data_directory + "/" + filename

    def load_movies(self) -> None:
        self.movies = pd.read_csv(self.__get_file(constants.MOVIES_METADATA))
        self.movies["genres"] = self.movies["genres"].fillna('[]').apply(literal_eval).apply(
            lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
        self.movies['production_companies'] = self.movies['production_companies'].fillna('[]').apply(
            literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
        self.movies['production_countries'] = self.movies['production_countries'].fillna('[]').apply(
            literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
        self.movies['spoken_languages'] = self.movies['spoken_languages'].fillna('[]').apply(
            literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
        self.movies['year'] = pd.to_datetime(self.movies['release_date'], errors='coerce').apply(
            lambda x: str(x).split('-')[0] if x != np.nan else np.nan)

    def __duplicate_for_genre(self) -> None:
        genre_serie = self.movies.apply(lambda x: pd.Series(
            x['genres']), axis=1).stack().reset_index(level=1, drop=True)
        genre_serie.name = "genre"
        self.movies_for_genres = self.movies.drop(
            "genres", axis=1).join(genre_serie)

    def process_data(self) -> None:
        if self.movies == None:
            self.load_movies()
        self.__duplicate_for_genre()

    def get_movie_by_category(self, category_name: str, percentile: float = .9) -> pd.DataFrame:
        if self.movies == None:
            print("Load movies first")
            exit(1)
        df = self.movies_for_genres[self.movies_for_genres['genre']
                                    == category_name]
        vote_counts = df[df['vote_count'].notnull()
                         ]['vote_count'].astype('int')
        vote_averages = df[df['vote_average'].notnull()
                           ]['vote_average'].astype('int')
        C = vote_averages.mean()
        m = vote_counts.quantile(percentile)

        qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & (
            df['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity']]
        qualified['vote_count'] = qualified['vote_count'].astype('int')
        qualified['vote_average'] = qualified['vote_average'].astype('int')

        qualified['wr'] = qualified.apply(
            lambda x: weighted_rating(x, C=C, m=m), axis=1)
        qualified = qualified.sort_values('wr', ascending=False).head(250)

        return qualified

    @staticmethod
    def load_model(directory_name: str = constants.PROCESSED_DATA_DIRECTORY) -> "MoviesDataProcessing":
        model = MoviesDataProcessing(data_directory=directory_name)
        model.movies = pd.read_csv(model.__get_file(constants.MOVIES_METADATA))
        return model

    def save_model(self, directory_name: str = constants.PROCESSED_DATA_DIRECTORY) -> None:
        if self.movies == None:
            print("We can't save model with empty dataframe")
            exit(1)
        self.movies.to_csv(directory_name + "/" + constants.MOVIES_METADATA, columns=[
                           'id', 'imdb_id', 'year', 'title', 'overview', 'homepage', 'popularity',
                           'runtime', 'spoken_languages', 'production_compagny', 'production_country',
                           'vote_average', 'vote_count'])
        print("Data saved")


def weighted_rating(df, C: float = None, m: float = None) -> float:
    v = df["vote_count"]
    R = df["vote_average"]
    if C == None:
        C = R.mean()
    if m == None:
        m = v.quantile(0.96)
    return v / (v+m) * R + m / (m+v) * C
