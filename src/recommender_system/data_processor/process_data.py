from ast import literal_eval

import numpy as np
import pandas as pd

from recommender_system import constants


def get_director(x):
    """Get the director name.
    Parameters :
    ------------
        x: row of series

    Returns :
    ---------
        string or NaN
    """
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


class ProcessData:
    """
        Class can :
            - load multiple files (links, movies, keywords, credits)
            - save it in one file without useless rows and columns
            - load the file for another usage
    """

    @staticmethod
    def save_model(data_directory: str = constants.DATA_DIRECTORY, number_of_lines_to_save: int = constants.NUMBER_OF_MOVIE_TO_SAVE,
                   order_by: str = constants.ORDER_BY_FOR_PROCESSED_MOVIES, ascending: bool = constants.ASCENDING_PROCESSED_DATA):
        """Static method to handle the data loading process.

        Parameters :
        ------------
            number_of_lines_to_save : int = constants.NUMBER_OF_MOVIE_TO_SAVE
                number of movies to save
            order_by : str = constants.ORDER_BY_FOR_PROCESSED_MOVIES
                column to order movies
            ascending : bool = constants.ASCENDING_PROCESSED_DATA
                ascending or not
        """
        model = ProcessData()
        model.__init(data_directory=data_directory)
        model.__load_data()
        model.__transform_data()
        model.__remove_null_and_duplicate_lines()
        model.__convert_data_to_other_types()
        model.__load_and_compute_links_file()
        model.__load_and_compute_keywords_and_credits()
        model.__convert_data_to_other_types2()
        model.__save_model(number_of_lines_to_save=number_of_lines_to_save,
                           order_by=order_by, ascending=ascending)
        return model

    @staticmethod
    def load_model(data_directory: str = constants.PROCESSED_DATA_DIRECTORY) -> "ProcessData":
        """Static method to handle the data loading process.

        Parameters :
        ------------
            data_directory : str = constants.PROCESSED_DATA_DIRECTORY
                directory of the data to load

        Returns :
        ---------
            model : ProcessData
                class used to manipulate the data
        """
        model = ProcessData()
        model.__init(data_directory=data_directory)
        model.__load_data()
        model.__transform_precessed_data()
        model.__convert_data_to_other_types()
        model.__convert_data_to_other_types2()
        return model

    def __init(self, data_directory: str = constants.DATA_DIRECTORY) -> None:
        """Private initialize method.

        Parameters :
        ------------
            data_directory : str = constants.DATA_DIRECTORY
                where are the datasets
        """
        if data_directory not in [constants.PROCESSED_DATA_DIRECTORY, constants.DATA_DIRECTORY]:
            print("error : directory not secure")
            exit(1)
        self.data_directory = data_directory
        self.is_processed_data = data_directory == constants.PROCESSED_DATA_DIRECTORY

    def __get_file(self, filename: str) -> str:
        """Get the filename.

        Parameters :
        ------------
            filename : str
                name of the file to load

        Returns :
        ---------
            path : str
                path of the file
        """
        return self.data_directory + "/" + filename

    def __load_data(self) -> None:
        """Loads the data."""
        movie_filename = constants.PROCESSED_MOVIES if self.is_processed_data else constants.MOVIES_METADATA
        self.movies = pd.read_csv(self.__get_file(
            movie_filename), low_memory=False)
        print("Successfully loaded data (shape :", self.movies.shape, ")")

    def __transform_data(self) -> None:
        """Transforms the columns and rows (to process).

        If a column contains [] -> we convert it into a python list
        We make some modifications to save only the useful data
        """
        self.movies['genres'] = self.movies.genres.fillna('[]').apply(literal_eval).apply(
            lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
        self.movies['production_companies'] = self.movies.production_companies.fillna('[]').apply(
            literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
        self.movies['production_countries'] = self.movies.production_countries.fillna('[]').apply(
            literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
        self.movies['spoken_languages'] = self.movies.spoken_languages.fillna('[]').apply(
            literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
        self.movies['year'] = pd.to_datetime(self.movies['release_date'], errors='coerce').apply(
            lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
        self.movies['tagline'] = self.movies.tagline.fillna('')
        self.movies['description'] = self.movies['overview'] + \
            self.movies['tagline']
        self.movies['description'] = self.movies.description.fillna(
            '').apply(lambda x: str.lower(x))
        print("Data successfully transformed")

    def __transform_precessed_data(self) -> None:
        """Transforms the loaded data."""
        self.movies['genres'] = self.movies['genres'].apply(literal_eval)
        self.movies['cast'] = self.movies['cast'].apply(literal_eval)
        self.movies['keywords'] = self.movies['keywords'].apply(literal_eval)

    def __remove_null_and_duplicate_lines(self) -> None:
        """Remove all null and duplicated lines for each columns we need
        and remove all useless columns.
        """
        self.movies.drop(
            inplace=True, index=self.movies.index[self.movies.id.duplicated()])
        self.movies.drop(
            inplace=True, index=self.movies.index[self.movies.imdb_id.duplicated()])
        self.movies.drop(
            inplace=True, index=self.movies.index[self.movies['status'] != 'Released'])
        self.movies.drop(
            inplace=True, index=self.movies.index[self.movies['release_date'].isnull()])
        self.movies.drop(
            inplace=True, index=self.movies.index[self.movies['imdb_id'].isnull()])
        self.movies.drop(
            inplace=True, index=self.movies.index[self.movies.imdb_id == '0'])
        self.movies.drop(
            inplace=True, index=self.movies.index[self.movies['popularity'].isnull()])
        self.movies.drop(
            inplace=True, index=self.movies.index[self.movies['title'].isnull()])
        self.movies.drop(
            inplace=True, index=self.movies.index[self.movies['year'] == 'NaT'])
        self.movies.drop(
            inplace=True, index=self.movies.index[self.movies['vote_count'].isnull()])
        self.movies.drop(
            inplace=True, index=self.movies.index[self.movies['vote_average'].isnull()])
        self.movies.drop(inplace=True, columns=["belongs_to_collection", "status", "budget",
                                                "original_title", "revenue", "video", "release_date", "adult",
                                                "homepage", "poster_path", "production_countries", "production_companies",
                                                "runtime", "spoken_languages"])
        print("Null and duplicates rows/columns successfully deleted -> shape :",
              self.movies.shape)

    def __convert_data_to_other_types(self) -> None:
        """Converts columns to other data types.
        Fill null data.
        """
        self.movies['year'] = self.movies['year'].astype('int')
        self.movies['vote_count'] = self.movies['vote_count'].astype('int')
        self.movies['vote_average'] = self.movies['vote_average'].astype(
            'float')
        self.movies['popularity'] = self.movies['popularity'].astype('float')
        self.movies['id'] = self.movies['id'].astype('int')
        self.movies['tagline'] = self.movies.tagline.fillna('')
        print("Columns successfully converted")

    def __load_and_compute_links_file(self) -> None:
        """Loads and computes the links file (to be processed)."""
        links = pd.read_csv(self.__get_file(
            constants.MOVIES_LINKS), low_memory=False)
        links.drop(inplace=True, index=links.index[links['imdbId'].isnull()])
        links.drop(inplace=True, index=links.index[links['tmdbId'].isnull()])
        links['movie_id'] = links['movieId']
        links.drop(inplace=True, columns=["movieId"])
        smd = self.movies[self.movies['id'].isin(links.tmdbId)]
        smd = pd.merge(smd, links.drop(columns=["imdbId"]).rename(
            columns={'tmdbId': 'id'}), on='id', how='left')
        smd.drop(inplace=True, index=smd.index[smd.imdb_id.duplicated()])
        self.movies = smd
        print("Links file successfully computed")

    def __load_and_compute_keywords_and_credits(self) -> None:
        """Loads and computes the keywords file (to be processed)."""
        credits = pd.read_csv(self.__get_file(
            constants.MOVIES_CREDITS), low_memory=False)
        keywords = pd.read_csv(self.__get_file(
            constants.MOVIES_KEYWORDS), low_memory=False)
        keywords['id'] = keywords['id'].astype('int')
        credits['id'] = credits['id'].astype('int')
        keywords.drop(
            inplace=True, index=keywords.index[keywords.id.duplicated()])
        credits.drop(
            inplace=True, index=credits.index[credits.id.duplicated()])
        self.movies = self.movies.merge(credits, on='id')
        self.movies = self.movies.merge(keywords, on='id')
        self.movies['cast'] = self.movies['cast'].apply(literal_eval)
        self.movies['crew'] = self.movies['crew'].apply(literal_eval)
        self.movies['keywords'] = self.movies['keywords'].apply(literal_eval)
        self.movies['director'] = self.movies['crew'].apply(get_director)
        self.movies['cast'] = self.movies['cast'].apply(
            lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
        self.movies['cast'] = self.movies['cast'].apply(
            lambda x: x[:4] if len(x) >= 4 else x)
        self.movies['keywords'] = self.movies['keywords'].apply(
            lambda x: [i['name'].replace("\xa0", "") for i in x] if isinstance(x, list) else [])
        self.movies.drop(inplace=True, columns=["crew"])
        print("Keywords and Casts files successfully computed")

    def __convert_data_to_other_types2(self) -> None:
        """Converts the new added columns."""
        self.movies['movie_id'] = self.movies['movie_id'].astype('int')
        print("Columns successfully converted")

    def __save_model(self, number_of_lines_to_save: int = constants.NUMBER_OF_MOVIE_TO_SAVE, order_by: str = constants.ORDER_BY_FOR_PROCESSED_MOVIES, ascending: bool = constants.ASCENDING_PROCESSED_DATA) -> None:
        """Saves the useful data in one file.

        Parameters :
        ------------
            number_of_lines_to_save : int = constants.NUMBER_OF_MOVIE_TO_SAVE
                number of movies to save
            order_by : str = constants.ORDER_BY_FOR_PROCESSED_MOVIES
                column to order movies
            ascending : bool = constants.ASCENDING_PROCESSED_DATA
                ascending or not
        """
        self.movies.sort_values(by=order_by, inplace=True, ascending=ascending)
        self.movies = self.movies.head(number_of_lines_to_save)
        self.movies.to_csv(constants.PROCESSED_DATA_DIRECTORY +
                           "/" + constants.PROCESSED_MOVIES)
        print("Data successfully saved (shape :", self.movies.shape, ")")
