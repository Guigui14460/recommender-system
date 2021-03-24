import re

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
        self.data = data.movies
        self.__convert_data(self.data, "keywords")
        self.__convert_data(self.data, "cast")
        self.__convert_data(self.data, "director")
        self.data.director = self.data.director.apply(lambda x: str.lower("".join(
            x.split(" "))).replace(".", "").replace("-", "").replace("'", "").replace(",", ""))
        self.__convert_data(self.data, "genres")
        print("Data converted")

        self.keywords_df = self.__concat_list_data_to_single_df(
            self.data.keywords, "keyword")
        self.cast_df = self.__concat_list_data_to_single_df(
            self.data.cast, "cast")
        self.directors_df = self.__concat_list_data_to_single_df(
            self.data.director, "director")
        self.genres_df = self.__concat_list_data_to_single_df(
            self.data.genres, "genre")
        print("Dataframes for TF-iDF vectorizer successfully created")

        self.cosine_sim = None
        if analyze:
            print("Analyze ...")
            self.__analyze(filename_output=filename)
        else:
            print("Load analyzed model ...")
            self.__load_model(filename_input=filename)
        print("Done !")

        self.titles = self.data['title']
        self.indices = pd.Series(self.data.index, index=self.data['title'])

    def recommend(self, title: str, nrows: int = None, **kwargs) -> pd.DataFrame:
        """Recommends movies based in content and metadata.

        Parameters:
        -----------
            title : str
                title of the movie
            nrows : int = None
                number of rows to return

        Returns:
        --------
            the recommended movies in a dataframe
        """
        idx = self.indices[title]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:21]
        movie_indices = [i[0] for i in sim_scores]
        movies = pd.DataFrame(
            self.data[self.data.title.isin(
                self.titles.iloc[movie_indices])][['movie_id', 'title']])
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
        data_for_tfidf = self.data.description + " " + \
            self.data.director + " " + self.data.cast
        data_for_tfidf += " " + self.data.keywords + " " + self.data.genres
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
                print(i)
        matrix[:, directors_indices] *= 3
        print("Directors weights computed")

        keywords_indices = []
        for i in self.keywords_df.keyword:
            try:
                keywords_indices.append(tf.get_feature_names().index(i))
            except:
                print(i)
        matrix[:, keywords_indices] *= 10
        print("Keywords weights computed")

        cast_indices = []
        for i in self.cast_df.cast:
            try:
                cast_indices.append(tf.get_feature_names().index(i))
            except:
                print(i)
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
