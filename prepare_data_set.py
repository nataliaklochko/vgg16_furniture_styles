import os
import numpy as np
from keras.preprocessing.image import load_img

script_dir = os.path.dirname(os.path.abspath(__file__))
data_set_dir = os.path.join(script_dir, 'data_set')
img_shape = (224, 224, 3)
set_size = len([0 for item in os.listdir(data_set_dir)])

styles_dict = {0: 'mid-century-modern', 1: 'modern', 2: 'art-deco',
		  	   3: 'scandinavian-modern', 4: 'neoclassical', 5:'victorian',
		  	   6: 'georgian', 7: 'hollywood-regency', 8: 'louis-xvi',
		  	   9: 'art-nouveau', 10: 'louis-xv', 11: 'regency',
		  	   12: 'industrial', 13: 'baroque', 14: 'folk-art',
		  	   15: 'empire', 16: 'rococo', 17: 'arts-and-crafts',
		  	   18: 'american-modern'}

# append vector to csv file
def append_vector(filename, n_class, vector, csvfile):
	csvfile.write("{},".format(filename))
	for coordinate in vector:
		csvfile.write("{},".format(coordinate))
	csvfile.write("{},".format(n_class))
	csvfile.write("\n")

def find_key(input_dict, value):
	return next((k for k, v in input_dict.items() if v == value), None)

with open('train_set.csv', 'a') as csv_train:
	with open('test_set.csv', 'a') as csv_test:
		num_train = 0
		for filename in os.listdir(data_set_dir):
			for style in styles_dict.values():
				if filename.endswith('.jpg') and filename.startswith(style):
					x = load_img(os.path.join(data_set_dir, filename),
						target_size=(img_shape[0], img_shape[1]))
					x = np.expand_dims(x, axis=0)
					img_vector = np.reshape(x, img_shape)
					if num_train < int(set_size * 0.8):
						append_vector(filename, find_key(styles_dict, style), img_vector, csv_train)
					else:
						append_vector(filename, find_key(styles_dict, style), img_vector, csv_test)
			num_train += 1
