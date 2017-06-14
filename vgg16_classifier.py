# using TensorFlow backend

from keras import applications
from keras import optimizers
from keras.models import Model
from keras.layers import Dropout, Dense, Flatten
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import np_utils
import numpy as np
import os

styles_dict = {0: 'mid-century-modern', 1: 'modern', 2: 'art-deco',
		  	   3: 'scandinavian-modern', 4: 'neoclassical', 5:'victorian',
		  	   6: 'georgian', 7: 'hollywood-regency', 8: 'louis-xvi',
		  	   9: 'art-nouveau', 10: 'louis-xv', 11: 'regency',
		  	   12: 'industrial', 13: 'baroque', 14: 'folk-art',
		  	   15: 'empire', 16: 'rococo', 17: 'arts-and-crafts',
		  	   18: 'american-modern'}

img_shape = (224, 224, 3)

nb_steps = 100
nb_epoch = 30
nb_classes = 19

# path to script and folders with images
script_dir = os.path.dirname(os.path.abspath(__file__))
data_set_dir = os.path.join(script_dir, 'data_set')
test_set_dir = os.path.join(script_dir, 'test_set')

set_size = len([0 for item in os.listdir(data_set_dir)])

train_set_file = 'train_set.csv'
test_set_file = 'test_set.csv'


def process_image(path, file):
	for i in range(nb_classes):
		if file.startswith(styles_dict[i]):
			if file.endswith('.jpg') or file.endswith('.jpeg'):
				x = load_img(os.path.join(path, file),
					target_size=(img_shape[0], img_shape[1]))
				x = np.expand_dims(x, axis=0)
				img_vector = np.reshape(x, (1,img_shape[0],img_shape[1],img_shape[2]))
				x = img_vector
				y = i
				x = x.astype('float32')
				x /= 255
				y = np_utils.to_categorical(y, nb_classes)
				return (x, y)

def generator(path):
    while 1:
        for file in os.listdir(path):
            x, y = process_image(path, file)
            yield (x, y)

model = applications.VGG16(weights="imagenet", include_top=False, input_shape=img_shape)

# freeze the layers which you don't want to train
for layer in model.layers[:14]:
	layer.trainable = False

# adding custom Layers 
x = model.output
x = Flatten()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(64, activation="relu")(x)
predictions = Dense(nb_classes, activation="softmax")(x)

# creating the final model 
model_final = Model(input = model.input, output = predictions)

# compile the model 
model_final.compile(loss="categorical_crossentropy", 
					optimizer=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999,
											  epsilon=1e-08, decay=0.0),
					metrics=["accuracy"])

print(model_final.summary())

# train the model
model_final.fit_generator(generator(data_set_dir), steps_per_epoch=nb_steps, epochs=nb_epoch)

# evaluate the accuracy
scores = model_final.evaluate_generator(generator(test_set_dir), steps=nb_steps)
print("Test accuracy: %.2f%%" % (scores[1] * 100))

# generate model description in json format
model_json = model_final.to_json()
# write model into file
json_file = open("vgg16_transfer_model.json", "w")
json_file.write(model_json)
json_file.close()
# save weights
model_final.save_weights("vgg16_transfer_weights.h5")
