from pandas import read_json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Engine():
	def __init__(self):
		self.data = read_json('assets/bdd.json') # data
		self.corpus = self.data['description'] # corpus
		self.vectorizer = TfidfVectorizer()
		self.X = self.vectorizer.fit_transform(self.corpus) # matrice de tf-idf

	def search(self, query, n=10): # query - mots-clés, n - nombre de résultats
		vq = self.vectorizer.transform([query]) # vectorisation de la requête
		vs = cosine_similarity(self.X, vq) # similarité entre le vecteur et la matrice de tf-idf
		self.data['similarite'] = vs # rajout d'une colonne 'similarite'
		df = self.data.sort_values(by='similarite', ascending=False).head(n) # trie du dataframe par ordre décroissant de tf-idf + retour de n premiers résultats
		res = "<table>"
		res += "<thead><tr><td>nom</td><td>réalisateur</td><td>année de sortie</td></tr></thead><tbody>"
		for index, row in df.iterrows():
			res += "<tr>"
			res += "<td>" + str(row['nom'])  + "</td>"
			res += "<td>" + str(row['realisateur'])  + "</td>"
			res += "<td>" + str(row['date'])  + "</td>"
			res += "</tr>"
		res += "</tbody></table>"
		return res