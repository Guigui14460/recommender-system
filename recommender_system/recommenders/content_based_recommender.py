import re
from typing import Union

import h5py
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from recommender_system.data_processor.process_data import ProcessData
from .recommender import Recommender

regex_parenthesis = re.compile(r"\(([a-zA-Z0-9]*)\)", re.IGNORECASE)
regex_double_quotes = re.compile(r"\"([a-zA-Z0-9]*)\"", re.IGNORECASE)


class ContentBasedRecommender(Recommender):
    """Class used to recommend movies based on movies content and metadata."""

    def __init__(self, data: ProcessData, filename: str, analyze: bool = False) -> None:
        """Initialize the instance.

        Parameters:
        -----------
            data : ProcessDate
                data which can use
            filename : str
                filename of the file to save or load the cosine similarity
            analyze : bool = False
                boolean to know if we need to analyze the data or not
        """
        super().__init__()
        self.data = data
        self.__convert_data(self.data.movies, "keywords")
        self.__convert_data(self.data.movies, "cast")
        self.__convert_data(self.data.movies, "director")
        self.data.movies.director = self.data.movies.director.apply(lambda x: str.lower("".join(
            x.split(" "))).replace(".", "").replace("-", "").replace("'", "").replace(",", ""))
        self.__convert_data(self.data.movies, "genres")
        print("Data converted")

        self.keywords_df = self.__concat_list_data_to_single_df(
            self.data.movies.keywords, "keyword")
        self.cast_df = self.__concat_list_data_to_single_df(
            self.data.movies.cast, "cast")
        self.directors_df = self.__concat_list_data_to_single_df(
            self.data.movies.director, "director")
        self.genres_df = self.__concat_list_data_to_single_df(
            self.data.movies.genres, "genre")
        print("Dataframes for TF-iDF vectorizer successfully created")

        self.cosine_sim = None
        if analyze:
            print("Analyze ...")
            self.__analyze(filename_output=filename)
        else:
            print("Load analyzed model ...")
            self.__load_model(filename_input=filename)
        print("Done !")

        self.titles = self.data.movies['title']
        self.indices = pd.Series(
            self.data.movies.index, index=self.data.movies['title'])

    def recommend(self, user_id: int, nrows: int = None, **kwargs) -> Union[pd.DataFrame, pd.Series]:
        """Recommends movies based in content and metadata.

        Parameters:
        -----------
            user_id : int
                title of the movie
            nrows : int = None
                number of rows to return

        Returns:
        --------
            the recommended movies in a dataframe
        """
        movie_ids = kwargs.get(
            "ratings", self.data.get_ratings_by_user_id(user_id))
        movies = self.data.movies.reset_index()
        movies['estimations'] = 0.0
        for i in movie_ids.index:
            idx = self.indices[movie_ids['title'][i]]
            similarity_score = np.array(
                list(enumerate(self.cosine_sim[idx])), dtype="object")
            if similarity_score[:, 0].shape < movies.index.values.shape:
                for j in range(similarity_score.shape[0]):
                    movies['estimations'] += similarity_score[:, 1][j]
            else:
                movies['estimations'] += similarity_score[:, 1]
        movies.sort_values(by="estimations", ascending=False, inplace=True)
        movies = movies[~movies.movie_id.isin(movie_ids.movie_id)]
        return movies if nrows == None else movies.head(nrows)

    def __convert_data(self, data: pd.DataFrame, column: str) -> None:
        """Converts the data.
        Remove all the useless characters (like ., ,, -, ', parenthesis).

        Parameters:
        -----------
            data : pd.DataFrame
                the main dataframe
            column : str
                column to modify
        """
        data[column] = data[column].apply(lambda x: [
            regex_double_quotes.sub(r' \g<1>',
                                    regex_parenthesis.sub(r" \g<1>", str.lower("".join(i.split(" "))).
                                                          replace(".", "").replace("-", "").
                                                          replace("'", "").replace(",", ""))) for i in x
        ]).apply(lambda x: " ".join(x))

    def __concat_list_data_to_single_df(self, serie: pd.Series, column: str) -> pd.Series:
        """Concatenates a serie for a particular column to a single dataframe.
        For exemple, if we have a serie of list of genres, this method
        concatenates all the value and remove duplicates.

        Parameters :
        ------------
            serie : pd.Series
                serie of the dataframe to concatenates
            column : str
                name of the column for the output dataframe

        Returns :
        ---------
            serie_frame : pd.Series
                new dataframe with the all concatenated data
        """
        serie_frame = pd.concat([pd.DataFrame(i, columns=[column])
                                 for i in serie.apply(lambda x: [i for i in x.split(" ")])
                                 ], ignore_index=True)
        serie_frame.drop(
            inplace=True, index=serie_frame.index[serie_frame[column].duplicated()])
        serie_frame.drop(
            inplace=True, index=serie_frame.index[serie_frame[column] == ""])
        return serie_frame

    def __analyze(self, filename_output: str) -> None:
        """Analyzes the movies data with a TF-iDF vectorizer.

        Parameters :
        ------------
            filename_output : str
                filename of the file where to save the cosine similarity
        """
        tf = TfidfVectorizer(analyzer='word', min_df=0, stop_words='english')
        data_for_tfidf = self.data.movies.description + " " + \
            self.data.movies.director + " " + self.data.movies.cast
        data_for_tfidf += " " + self.data.movies.keywords + " " + self.data.movies.genres
        tfidf_matrix = tf.fit_transform(data_for_tfidf)
        self.__put_weights(tfidf_matrix, tf)
        # similarity of one movies compare to others movies
        self.cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        with h5py.File(filename_output, "w") as f:
            f.create_dataset("cosime_sim", data=self.cosine_sim)

    def __put_weights(self, matrix: np.ndarray, tf: TfidfVectorizer) -> None:
        """Puts weights to some columns for the cosine similarity calculus.

        Parameters:
        -----------
            matrix : np.nd.array
                TF-iDF computed matrix
            tf : TfidfVectorizer
                vectorizer for the calculus of Tf-iDF
        """
        genres_indices = []
        for i in self.genres_df.genre:
            genres_indices.append(tf.get_feature_names().index(i))
        matrix[:, genres_indices] *= 5
        print("Genres weights computed")

        directors_indices = []
        for i in self.directors_df.director:
            try:
                directors_indices.append(tf.get_feature_names().index(i))
            except:
                print("Director not found :", i)
        matrix[:, directors_indices] *= 3
        print("Directors weights computed")

        keywords_indices = []
        for i in self.keywords_df.keyword:
            try:
                keywords_indices.append(tf.get_feature_names().index(i))
            except:
                print("Keyword not found :", i)
        matrix[:, keywords_indices] *= 10
        print("Keywords weights computed")

        cast_indices = []
        for i in self.cast_df.cast:
            try:
                cast_indices.append(tf.get_feature_names().index(i))
            except:
                print("Cast not found :", i)
        matrix[:, cast_indices] *= 3
        print("Casts weights computed")

    def __load_model(self, filename_input: str) -> None:
        """Loads the model saved in H5 file.

        Parameters :
        ------------
            filename_input : str
                filename of the file used to loads the cosine similarity between movies
        """
        with h5py.File(filename_input, "r") as f:
            self.cosine_sim = np.array(f["cosime_sim"])
