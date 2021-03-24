import pandas as pd
import numpy as np

from nltk.corpus import wordnet
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate

from recommender_system.data_processor.process_data import ProcessData
from .recommender import Recommender
from .content_based_recommender import ContentBasedRecommender


class CombinationRecommender (Recommender):
    def __init__(self, data: ProcessData, filename: str) -> None:
        super().__init__()
        self.data = data.movies
        self.ratings = pd.read_csv('data/ratings_small.csv')
        self.id_map = pd.read_csv('data/links.csv')[['movieId', 'tmdbId']]
        self.id_map.columns = ['movieId', 'id']
        self.id_map = self.id_map.merge(
            self.data[['title', 'id']], on='id').set_index('title')
        self.indices_map = self.id_map.set_index('id')
        self.content_based_recommender = ContentBasedRecommender(
            data, filename)
        self.svd = SVD()
        self.rating_user = self.ratingUser()
        print("Combined initialized")

    def ratingUser(self):
        reader = Reader()
        data = Dataset.load_from_df(
            self.ratings[['userId', 'movieId', 'rating']], reader)
        cross_validate(self.svd, data, measures=['RMSE', 'MAE'], cv=5)
        trainset = data.build_full_trainset()
        print("Training the SVD ...")
        self.svd.fit(trainset)
        print("SVD ready !")
        user_ratings = pd.merge(
            self.ratings, self.data, left_on='movieId', right_on='id', how='inner')
        user_ratings_final = user_ratings[[
            'userId', 'movieId', 'rating', 'title']]
        user_ratings = user_ratings_final.sort_values(by='userId')

        return user_ratings

    def recommend(self, userId, **kwargs):
        allMoviesUser = self.rating_user[self.rating_user['userId'] == userId]
        moviesRecommender = []
        moviesTitle = []
        moviesId = []
        for i in allMoviesUser.iterrows():
            idx = i[1]['title']
            moviesTitle.append(idx)
            moviesId.append(i[1]['movieId'])
        for j in moviesTitle:
            movies = self.data[self.data.movie_id.isin(
                               self.content_based_recommender.recommend(j).movie_id)]
            movies['est'] = movies.movie_id.apply(lambda x: self.svd.predict(
                userId, x).est)
            movies.sort_values('est', ascending=False, inplace=True)
            movies['movie'] = j
            moviesRecommender.append(movies.head(5))
        res = pd.concat(moviesRecommender).drop_duplicates(subset=['id'])
        return res[~res.id.isin(moviesId)]
