###Extract ingredients from text
#from intent_extraction import *

quantity = ['tablespoon', 'tablespoons', 'optional', 'cup', 'cups']

def extract_ingredients():
	f = open('Recipes.txt', 'r', encoding='utf-8')
	data = f.readlines()
	#print(data[1])
	for line in data:
		ingredients = []
		a = ' '.join(line.split(',')[2:])
		ings = a.split('[')[1].split('  ')
		print(ings)
		for ing in ings:
			ing
			ingredients.append()
		#for ing in ingredients:
		#	np_extractor = NPExtractor(ing)
		#	result = np_extractor.extract()
		#	print(result)

		break

extract_ingredients()