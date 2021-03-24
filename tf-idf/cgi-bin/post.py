#coding: utf-8

import cgi
import os
import sys # pour gérer les accents

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import Engine

def enc_print(string='', encoding='utf8'): # pour gérer les accents
	sys.stdout.buffer.write(string.encode(encoding) + b'\n')

enc_print("Content-type: text/html; charset=utf-8\n")

form = cgi.FieldStorage()
if form.getvalue("query") and form.getvalue("nb_results"):
	query = form.getvalue("query")
	nb_results =  int(form.getvalue("nb_results")) if int(form.getvalue("nb_results")) >= 1 else 1
	html = """<!DOCTYPE html>
<html>
	<head>
		<meta charset="utf-8">
		<title>Système de recommandation - résultats</title>
        <link href="../assets/style.css" rel="stylesheet">
	</head>
	<body>
		{}
	</body>
</html>
""".format(Engine().search(query, nb_results))
	enc_print(html)
else:
	enc_print("Vous n'avez pas rempli tous les champs")