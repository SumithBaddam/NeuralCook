# API for food image classification
import flask
from flask import request, jsonify
import requests
import json

#% matplotlib inline
import matplotlib
import numpy as	np
import h5py
import os
import matplotlib.pyplot as	plt
import numpy as	np
import pandas as pd
import ast
import cv2
from keras.utils.io_utils import HDF5Matrix
import numpy as	np

import keras
from keras.preprocessing import	image
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, MaxPool2D, Flatten, Dense, Dropout, Activation, Input
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import	KerasClassifier
from sklearn.preprocessing import LabelEncoder,	OneHotEncoder, LabelBinarizer

from keras.applications	import ResNet50
from keras.applications	import InceptionV3
from keras.applications	import Xception
from keras.applications	import VGG16, VGG19, InceptionV3
from keras.applications	import imagenet_utils
from keras.applications.inception_v3 import	preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import	ModelCheckpoint, LearningRateScheduler,	TensorBoard, EarlyStopping 
from keras.models import model_from_json

from datetime import timedelta
from flask import make_response, request, current_app
from functools import update_wrapper

app	= flask.Flask(__name__)
app.config["DEBUG"]	= True

def	crossdomain(origin=None, methods=None, headers=None, max_age=21600,
				attach_to_all=True,	automatic_options=True):

	if methods is not None:
		methods	= ', '.join(sorted(x.upper() for x in methods))
	if headers is not None and not isinstance(headers, list):
		headers	= ', '.join(x.upper() for x	in headers)
	if not isinstance(origin, list):
		origin = ',	'.join(origin)
	if isinstance(max_age, timedelta):
		max_age	= max_age.total_seconds()

	def	get_methods():
		#Determines which methods are allowed
		if methods is not None:
			return methods

		options_resp = current_app.make_default_options_response()
		return options_resp.headers['allow']

	def	decorator(f):
		#The decorator function
		def	wrapped_function(*args,	**kwargs):
			#Caries out the actual cross domain code
			if automatic_options and request.method	== 'OPTIONS':
				resp = current_app.make_default_options_response()
			else:
				resp = make_response(f(*args, **kwargs))
			if not attach_to_all and request.method	!= 'OPTIONS':
				return resp

			h =	resp.headers
			h['Access-Control-Allow-Origin'] = origin
			h['Access-Control-Allow-Methods'] =	get_methods()
			h['Access-Control-Max-Age']	= str(max_age)
			h['Access-Control-Allow-Credentials'] =	'true'
			h['Access-Control-Allow-Headers'] =	\
				"Origin, X-Requested-With, Content-Type, Accept, Authorization"
			if headers is not None:
				h['Access-Control-Allow-Headers'] =	headers
			return resp

		f.provide_automatic_options	= False
		return update_wrapper(wrapped_function,	f)
	return decorator


image_dir =	'public/food-101/images'
image_size = (224, 224)
batch_size = 16

train_datagen =	ImageDataGenerator(rescale = 1./255,horizontal_flip	= False,fill_mode =	"nearest",zoom_range = 0,
								   width_shift_range = 0,height_shift_range=0,rotation_range=0)

train_generator	= train_datagen.flow_from_directory(image_dir,target_size =	(image_size[0],	image_size[1]),
													batch_size = batch_size, class_mode	= "categorical", shuffle=False)

train_generator.reset()
num_of_classes = len(train_generator.class_indices)


# load json	and	create model
json_file =	open('public/model_final.json',	'r')
loaded_model_json =	json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into	new	model
loaded_model.load_weights("public/model_final.h5")
loaded_model._make_predict_function()
print("Loaded model	from disk")


#Take image	name as	input and remove everything	else from running on model
@app.route('/classification', methods=['GET', 'OPTIONS'])
@crossdomain(origin='*')
def	api_classification():
	imgname	= ''
	predicted_answer = None
	if('image_name'	in request.args):
		imgname	= request.args['image_name'].split('/')[-1]
	print('Image name:', imgname, '	: ', request.args['image_name'])
	if 'image_path'	in request.args:
		img_path = request.args['image_path']
		print(img_path)

		#Read from file
		train_generator2 = train_datagen.flow_from_directory(img_path, target_size = (image_size[0], image_size[1]),
													batch_size = 1,	class_mode = "categorical",	shuffle	= False)
		train_generator2.reset()
		actuals	= train_generator2.filenames
		print(actuals)
		#Classification
		for	i in range(len(actuals)):
			_ =	train_generator2.next()
			image, classifier =	(_[0][0],_[1][0])
			#print(image.shape)
			#if(actuals[i] == imgname):
			if(i ==	len(actuals)-1):
				predicted =	loaded_model.predict(np.asarray([image]))
				predicted_answer_index = np.argmax(predicted[0])
				predicted_answer = list(train_generator.class_indices.keys())[predicted_answer_index]
				#plt.imshow(image)
				#plt.show()
				print('Prediction:', predicted_answer)
		
		return json.dumps({'predicted':	predicted_answer})

	else:
		return "Error: Image not found in specific path."


similarity_model = VGG16(weights='imagenet', include_top=True)
feat_extractor = Model(input=similarity_model.input, output=similarity_model.get_layer("fc2").output)
feat_extractor._make_predict_function()


@app.route('/similarity', methods=['GET', 'OPTIONS'])
@crossdomain(origin='*')
def	api_similarity():
	if(('image_path' in	request.args) and ('classes' in	request.args)):
		img_path = request.args['image_path']
		classes	= request.args['classes']

		#Read from file
		train_generator2 = train_datagen.flow_from_directory(img_path, target_size = (image_size[0], image_size[1]),
													batch_size = 1,	class_mode = "categorical",	shuffle	= False)
		train_generator2.reset()
		actuals	= train_generator2.filenames
		print('fetching	query image')
		#Get embedding for query image
		for	i in range(len(actuals)):
			_ =	train_generator2.next()
			image, classifier =	(_[0][0],_[1][0])
			if(i ==	len(actuals)-1):
				query =	feat_extractor.predict(np.asarray([image]))[0]

		#Read extracted	features from file
		classes	= classes.split(',')

		print('fetching	embeddings')
		features = []
		image_names=[]
		for	category in	classes:
			f =	open('public/food-101/image_embeddings/' + category	+ '/' +'image_embedding.txt', encoding='utf-8')
			for	a in f.readlines()[:20]:
				if(a.split(',')[0].split('\\')[0] in classes):
					features.append(ast.literal_eval(','.join(a.split(',')[1:]).strip()))
					image_names.append(a.split(',')[0])

		print('Computing similarity')
		distances =	[]
		for	i in range(len(features)):
			dot	= np.dot(query,	features[i])
			norma =	np.linalg.norm(query)
			normb =	np.linalg.norm(features[i])
			cos	= dot /	(norma * normb)
			distances.append(cos)

		print('computing results')
		dist_indices = sorted(range(len(distances)), key=lambda	k: distances[k])[-9:][::-1]
		#print(dist_indices)
		result = ''
		sim_scores = ''
		for	i in dist_indices:
			result = result	+ image_names[i].replace('\\', '/')	+ ','
			sim_scores = sim_scores	+ str(distances[i])	+ ','
		print(sim_scores, result)
		a =	json.dumps({'similarImages': result.strip(','), 'confidence_scores': sim_scores.strip(',')})
		print(a)
		return a

	return



f =	open('ingredients.txt')
all_ingredients	= f.readlines()

ingredients	= {}
for	i in all_ingredients:
	ingredients[i.split(',')[0]] = ', '.join(i.split(',')[1:]).strip()


with open('ingr_predictions.json') as f:
	ingr_predictions = json.load(f)

for	i in list(ingr_predictions.keys()):
	ingr_predictions[i]	= ast.literal_eval(ingr_predictions[i])


@app.route('/imagetorecipe', methods=['GET', 'OPTIONS'])
@crossdomain(origin='*')
def	api_recipe():
	if(('image_path' in	request.args) and ('classes' in	request.args)):
		img_path = request.args['image_path']
		classes	= request.args['classes']

		#Read from file
		train_generator2 = train_datagen.flow_from_directory(img_path, target_size = (image_size[0], image_size[1]),
													batch_size = 1,	class_mode = "categorical",	shuffle	= False)
		train_generator2.reset()
		actuals	= train_generator2.filenames
		print('fetching	query image')
		#Get embedding for query image
		for	i in range(len(actuals)):
			_ =	train_generator2.next()
			image, classifier =	(_[0][0],_[1][0])
			if(i ==	len(actuals)-1):
				query =	feat_extractor.predict(np.asarray([image]))[0]


		similarity_scores =	[]
		image_names	= []
		for	i in range(len(list(ingr_predictions.keys()))):
			a =	list(ingr_predictions.keys())[i]
			if(a.split('\\')[0]	in classes):
				ingr = ingr_predictions[a]
				# manually compute cosine similarity
				dot	= np.dot(query,	ingr)
				norma =	np.linalg.norm(query)
				normb =	np.linalg.norm(ingr)
				cos	= dot /	(norma * normb)
				similarity_scores.append(cos)
				image_names.append(a)


		#Get top 10	recipe predictions
		result = ''
		top_recipe_idx = sorted(range(len(similarity_scores)), key=lambda i: similarity_scores[i])[-10:]
		for	idx	in top_recipe_idx[::-1]:
			print(image_names[idx],	similarity_scores[idx])
			result = result	+ image_names[idx] +','
		print(result)
		a =	json.dumps({'recipe': ingredients[result.split(',')[0]]})
		return a


#Fetching ingredient
def	compute_similarity(list1, list2):
	idx	= []
	for	i in list1:
		status = False
		for	j in i.split(' '):
			#print(j)
			for	k in range(len(list2)):
				if(j.strip() in list2[k]):
					status = True
					idx.append(k)
					break
			if(status == True):
				break
	#print(idx)
	missing_ingr = []
	for i in range(len(list2)):
		if(i not in idx):
			missing_ingr.append(list2[i])

	#Check for all ingredients present
	return (float(len(idx)/len(list2)), ','.join(missing_ingr))


@app.route('/ingredientstoimage', methods=['GET', 'OPTIONS'])
@crossdomain(origin='*')
def	api_ingredientstoimage():
	if('ingredients' in	request.args):
		query =	request.args['ingredients'].split(',')
		print(query)
		#Run ingr_prediction and compare with those	vectors
		#Run ingr_prediction with images

		#Words for confidence
		similarity_score = []
		missing_ingr = []
		image_names	= list(ingredients.keys())
		for	i in image_names:
			ingr = ingredients[i].split(', ')
			a, b = compute_similarity(query, ingr)
			similarity_score.append(a)
			missing_ingr.append(b)

		dist_indices = sorted(range(len(similarity_score)),	key=lambda k: similarity_score[k])[-9:][::-1]
		result = sim_scores	= missing = ''
		#Put a filter on the threshold...
		for	i in dist_indices:
			result = result	+ image_names[i].replace('\\', '/')	+ ','
			sim_scores = sim_scores	+ str(similarity_score[i])	+ ','
			missing = missing + missing_ingr[i] + '::'

		print(sim_scores)
		print(result)
		print(missing)
		a =	json.dumps({'images': result.strip(','), 'confidence_scores': sim_scores.strip(','), 'missing_ingredients': missing.strip('::')})
		return a


@app.route('/recipe', methods=['GET', 'OPTIONS'])
@crossdomain(origin='*')
def	api_recipefromimage():
	if('item' in request.args):
		item =	request.args['item']
		#print(ingredients.keys())
		a = json.dumps({'ingredients': ingredients[item.replace('/','\\')]})
		#print(a)
		return a


app.run(port=3001)
