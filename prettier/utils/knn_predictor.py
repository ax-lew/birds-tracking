from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor



class KnnPredictor(object):
	"""docstring for KnnPredictor"""
	def __init__(self):
		super(KnnPredictor, self).__init__()
		self.clf = MultiOutputRegressor(KNeighborsRegressor(n_neighbors=5, weights='distance'))

	def fit(self, data, target):
		self.clf.fit(data, target)

	def predict(self, data):
		return self.clf.predict(data)		
		
