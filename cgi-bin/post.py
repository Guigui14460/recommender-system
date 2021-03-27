#coding: utf-8

import cgi
import os
import sys # pour gérer les accents
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recommender_system.data_processor import ProcessData
from recommender_system.recommenders import CombinationRecommender

data = ProcessData.load_model()
recommender = CombinationRecommender(data, "processed_movies_similarity.h5", "trained_svd_model.pickle")

def enc_print(string='', encoding='utf8'): # pour gérer les accents
	sys.stdout.buffer.write(string.encode(encoding) + b'\n')

enc_print("Content-type: text/html; charset=utf-8\n")

form = cgi.FieldStorage()
if form.getvalue("user_id") and form.getvalue("nb_results"):
	user_id = int(form.getvalue("user_id")) if 1 <= int(form.getvalue("user_id")) <= 671 else 1
	nb_results =  int(form.getvalue("nb_results")) if int(form.getvalue("nb_results")) >= 1 else 1
	cols = ['title', 'movie_id', 'estimations']
	recommendations = recommender.recommend(user_id, nrows=nb_results)[cols].to_html(columns=cols, bold_rows=False, index=False)
	html = """<!DOCTYPE html>
<html>
	<head>
		<meta charset="utf-8">
		<title>Système de recommandation - résultats</title>
        <link href="../assets/style.css" rel="stylesheet">
	</head>
	<body>
    	<h2>Films vus et notés par l'utilisateur {}</h2>
        {}
    	<h2>{} films recommandés à l'utilisateur {}</h2>
		{}
	</body>
</html>
""".format(user_id, recommender.data.get_ratings_by_user_id(user_id).to_html(columns=['title', 'movie_id', 'rating'], bold_rows=False, index=False), nb_results, user_id, recommendations)
	enc_print(html)
else:
	enc_print("Vous n'avez pas rempli tous les champs")