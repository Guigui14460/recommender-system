#coding: utf-8

import cgi
import sys # pour gérer les accents

def enc_print(string='', encoding='utf8'): # pour gérer les accents
	sys.stdout.buffer.write(string.encode(encoding) + b'\n')

enc_print("Content-type: text/html; charset=utf-8\n")

html = """<!DOCTYPE html>
<html>
	<head>
		<meta charset="utf-8">
		<title>Système de recommandation</title>
        <link href="../assets/style.css" rel="stylesheet">
	</head>
	<body>
		<h1>Système de recommandation</h1>
		
		<form accept-charset="utf-8" method="post" action="post.py">
			<p>
                <label for="query">Qu'est-ce qui vous intéresse ?</label><br>
				<input type="text" name="query" id="query" placeholder="Tapez des mots-clés..."><br>
                <label for="nb_results">Combien de films voulez-vous avoir en recommandation ? (1+)</label><br>
                <input type="number" name="nb_results" id="nb_results" min="1" value="5"><br>
				<input type="submit" value="Afficher les recommandations">
			</p>
		</form>
	</body>
</html>
"""

enc_print(html)