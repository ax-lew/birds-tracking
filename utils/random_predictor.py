from random import randint
import numpy as np


class RandomPredictor(object):
	"""docstring for RandomPredictor"""
	def __init__(self):
		super(RandomPredictor, self).__init__()

	def fit(self, data, target):
		self.max_x, self.max_y = target.max()
		self.min_x, self.min_y = target.min()

	def predict(self, data):
		res = []
		for i in range(len(data)):
			res.append((randint(int(self.min_x), int(self.max_x)), randint(int(self.min_y), int(self.max_y))))
		return np.array(res)
		
