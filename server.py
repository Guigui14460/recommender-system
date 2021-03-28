from flask import Flask, request, render_template, redirect, url_for

from recommender_system.data_processor import ProcessData
from recommender_system.recommenders import CombinationRecommender


data = ProcessData.load_model()
recommender = CombinationRecommender(
    data, "processed_movies_similarity.h5", "trained_svd_model.pickle")

app = Flask(__name__)
app.config.update(
    TESTING=True,
)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', error=None)


@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    if request.method == "GET":
        return redirect(url_for('index'))
    user_id = request.form.get('user_id')
    nb_results = request.form.get('nb_results')
    if user_id is None or nb_results is None:
        return render_template("index.html", error="Il manque au moins une des deux valeurs")
    user_id = int(user_id)
    if user_id < 1 or user_id > 671:
        return render_template("index.html", error="L'identifiant utilisateur doit se trouver entre 1 et 671")
    nb_results = int(nb_results)
    if nb_results < 1:
        nb_results = 1
    cols = ['title', 'movie_id', 'estimations']
    recommended_movies = recommender.recommend(user_id, nrows=nb_results)[
        cols].to_html(columns=cols, bold_rows=False, index=False)
    rated_movies = recommender.data.get_ratings_by_user_id(user_id).to_html(
        columns=['title', 'movie_id', 'rating'], bold_rows=False, index=False)
    return render_template("recommend.html", user_id=user_id, recommended_movies=recommended_movies, rated_movies=rated_movies)


if __name__ == '__main__':
    app.run()
