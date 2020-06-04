import pandas as pd
from datetime import datetime, timedelta
import numpy as np

class Smoother(object):
	"""docstring for EmissionsNormalizer"""
	def __init__(self, windows_size=60, step_size=10, window_threshold=500, use_median=True):
		self.windows_size = windows_size
		self.step_size = step_size
		self.window_threshold = window_threshold
		self.use_median = use_median


	def smooth_predictions(self, predictions):
		assert len(predictions[0]) == 8

		smooth_predictions = []
		window_predictions = []

		init_time_map = {}
		for i in range(len(predictions)):
			prediction = predictions[i]
			bird_id = prediction[6]
			init_time = init_time_map.get(bird_id)
			if init_time is not None:
				new_init_time = datetime.strptime(prediction[7], '%Y-%m-%d %H:%M:%S')
				delta = new_init_time - init_time
				if delta.total_seconds() <= self.step_size:
					continue
			window_predictions = []
			window_predictions.append(prediction)
			init_time = datetime.strptime(prediction[7], '%Y-%m-%d %H:%M:%S')
			init_time_map[bird_id] = init_time
			for j in range(i+1, len(predictions)):
				next_prediction = predictions[j]
				if bird_id != next_prediction[6]:
					continue
				time_diff = datetime.strptime(next_prediction[7], '%Y-%m-%d %H:%M:%S') - init_time
				if time_diff.total_seconds() <= self.windows_size:
					window_predictions.append(next_prediction)
				else:
					break
			window_predictions_np = np.array(window_predictions)
			if self.use_median:
				x = np.median(window_predictions_np[:,0])
				y = np.median(window_predictions_np[:,1])
			else:
				x = window_predictions_np[:,0].mean()
				y = window_predictions_np[:,1].mean()
			smooth_predictions.append([x, y, None, None, None, None, bird_id, window_predictions[0][7]])

		return np.array(smooth_predictions)

	def filter_predictions(self, predictions):
		predictions_pd = pd.DataFrame(predictions, columns=['x', 'y', 'recep_0', 'recep_1', 'recep_2', 'recep_3', 'tag', 'time'])
		predictions_pd['time'] = pd.to_datetime(predictions_pd['time'], format='%Y-%m-%d %H:%M:%S')

		filtered_predictions = []
		for i in range(len(predictions)):
			prediction = predictions[i]
			bird_id = prediction[6]
			prediction_time = datetime.strptime(prediction[7], '%Y-%m-%d %H:%M:%S')
			init_time = prediction_time-timedelta(seconds=int(self.windows_size/2))
			end_time = prediction_time+timedelta(seconds=int(self.windows_size/2))
			
			window_predictions = predictions_pd[(predictions_pd.tag == bird_id) & (predictions_pd.time >= init_time) & (predictions_pd.time < end_time)]

		   
			if self.use_median:
				window_x = window_predictions.x.median()
				window_y = window_predictions.y.median()
			else:
				window_x = window_predictions.x.mean()
				window_y = window_predictions.y.mean()
			if abs(prediction[0]-window_x) < self.window_threshold and abs(prediction[1]-window_y) < self.window_threshold:
				filtered_predictions.append(prediction)

		return np.array(filtered_predictions)
			  