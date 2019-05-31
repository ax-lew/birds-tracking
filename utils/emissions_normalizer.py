import pandas as pd

class EmissionsNormalizer(object):
	"""docstring for EmissionsNormalizer"""
	def __init__(self):
		super(EmissionsNormalizer, self).__init__()
		self.x_0 = 462385.503783397
		self.y_0 = 6109042.35153865

	
	def _normalize_positions(self, row):
	    row['x'] = row['x']-self.x_0
	    row['y'] = row['y']-self.y_0
	    return row


	def _group_emisions(self, data):
	    data_dict = data.asDict()
	    recep_0 = data_dict.pop('recep_0')
	    recep_1 = data_dict.pop('recep_1')
	    recep_2 = data_dict.pop('recep_2')
	    recep_3 = data_dict.pop('recep_3')
	    
	    data_dict['emissions'] = []
	    for i in range(24):
	        data_dict['emissions'].append([recep_0[i], recep_1[i], recep_2[i], recep_3[i]])
	    return data_dict

	def _expand_rows_with_emissions(self, row):
	    emissions = row.pop('emissions')
	    rows = []
	    for e in emissions:
	        new_row = row.copy()
	        new_row['recep'] = e
	        rows.append(new_row)
	    return rows

	def _generate_attrs(self, row):
	    data = {
	        'antenna_0': row['recep'][0],
	        'antenna_1': row['recep'][1],
	        'antenna_2': row['recep'][2],
	        'antenna_3': row['recep'][3],
	    }
	    return {'data': data, 'x': row['x'], 'y': row['y'], 'point': row['Punto']}

	def normalize(self, points_recep):
		points_emisions = points_recep.map(self._group_emisions)
		all_emissions = (
			points_emisions
			.flatMap(self._expand_rows_with_emissions)
			.map(self._generate_attrs)
			.filter(lambda x: sum(x['data'].values())>0)
			.map(self._normalize_positions)
		)

		return all_emissions
	
	def get_regression_dataframes(self, all_emissions):
		regre_target = pd.DataFrame(all_emissions.map(lambda x: [x['x'], x['y']]).collect())
		regre_data = pd.DataFrame(all_emissions.map(lambda x: x['data']).collect())
		return regre_data, regre_target



		






    
		

