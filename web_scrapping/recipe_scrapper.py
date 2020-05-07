# Web scrapping recipes
#import urllib2
import urllib.request as urllib2
import json
import collections, itertools
import requests
from bs4 import BeautifulSoup
import csv
import requests
import shutil
import os
#https://www.allrecipes.com/recipe/195042/island-style-fried-rice/?internalSource=hub%20recipe&referringContentType=Search&clickId=cardslot%203
#https://www.allrecipes.com/recipe/7565/too-much-chocolate-cake/?internalSource=hub%20recipe&referringContentType=Search&clickId=cardslot%202

def scrap(dish, f, url, total):
	try:
		#req = urllib2.Request(url)
		#response = urllib2.urlopen(req)
		#print(response)
		#news_results = json.loads(response.read())
		#URL = 'https://www.monster.com/jobs/search/?q=Software-Developer&where=Australia'

		page = requests.get(url)
		soup = BeautifulSoup(page.content, 'html.parser')
		#print(soup.find(id='grid-card-image-container').preffify)
		#a = soup.find('article').find("div", {'class': 'fixed-recipe-card'}).find_all('p')
		#f = open('Recipes.csv', 'w+')

		all_items = soup.find_all("article", {'class': 'fixed-recipe-card'})
		#print(all_items)
		c = total
		for item in all_items:
			top_item = item.find("div", {'class': 'fixed-recipe-card__info'}).find("a", {'class': 'fixed-recipe-card__title-link'})#.find("a", href=True)
			dish_url = top_item['href']
			print(dish_url)
			
			page = requests.get(dish_url)
			soup = BeautifulSoup(page.content, 'html.parser')
			'''
			#image_url = soup.find_all("div", {"class": "inner-container js-inner-container image-overlay"})
			image_url = soup.find_all("img")
			print(image_url)
			for img in image_url:
				if("alt" in img):
					if(img["alt"] != ""):
						print(img)
			'''
			if(soup.find("h1", {'class': 'headline heading-content'})):
				#Get image url
				title = soup.find("h1", {"class": "headline heading-content"}).text
				author = soup.find("a", {"class": "author-name link"})
				if(author):
					author = author.text
				else:
					author = soup.find("span", {"class": "author-name"}).text
				alt = title + " " + author
				image_url = soup.find("img", {"alt": alt})

				#Save image
				if(image_url):
					save_image(dish, image_url["src"], c)

				#Fetch ingredients
				ingredients = []
				web_ingredients = soup.find_all("span", {'class': 'ingredients-item-name'})
				for ing in web_ingredients:
					ingredients.append(ing.text.strip())
				
				#print(ingredients)

				#Fetch recipe
				recipe_steps = []
				web_recipe = soup.find('ul', {'class': 'instructions-section'}).find_all('p')
				for rec in web_recipe:
					recipe_steps.append(rec.text)

				#print(recipe_steps)
				result = str(dish) + "," + str(dish_url) + "," + str(ingredients) + "," + str(recipe_steps) + "\n"
				f.write(result)

			
			elif(soup.find("h1", {'id': 'recipe-main-content'})):
				#Get image url
				title = soup.find("h1", {"class": "recipe-summary__h1"}).text
				author = soup.find("span", {"class": "submitter__name"}).text
				alt = "Photo of " + title + " by " + author
				image_url = soup.find("img", {"alt": alt})

				#Save image
				if(image_url):
					save_image(dish, image_url["src"], c)

				#Fetch ingredients
				ingredients = []
				web_ingredients = soup.find_all("li", {'class': 'checkList__line'})
				for ing in web_ingredients:
					if(len(ing.text.strip()) > 0 and ing.text.strip() != "Add all ingredients to list"):
						ingredients.append(ing.text.strip())
				
				#print(ingredients)

				#Fetch recipe
				recipe_steps = []
				web_recipe = soup.find_all('span', {'class': 'recipe-directions__list--item'})
				for rec in web_recipe:
					if(len(rec.text) > 0):
						recipe_steps.append(rec.text.strip())

				#print(recipe_steps)
				result = str(dish) + "," + str(dish_url) + "," + str(ingredients) + "," + str(recipe_steps) + "\n"
				f.write(result)

			c += 1

	except urllib2.HTTPError:
		return("wrong url")
	return c


def save_image(dish, image_url, num):
	resp = requests.get(image_url, stream=True)
	if not os.path.exists("data/" + dish):
		os.makedirs("data/" + dish)
	local_file = open("data/" + dish + "/" + str(num) + ".jpg", 'wb')
	resp.raw.decode_content = True
	shutil.copyfileobj(resp.raw, local_file)
	del resp
	return


def main(page_count):
	file = open("classes.txt", "r")
	f = open('Recipes.txt', 'a', encoding='utf-8')
	all_classes = file.read().split()
	count = 0
	for dish in all_classes:
		total = 0
		count += 1
		print("\n\n\n\n\n" + str(count) + "\n\n\n\n")
		for i in range(1, page_count+1):
			url = "https://www.allrecipes.com/search/results/?wt=" + '%20'.join(dish.split('_'))  + "&sort=re" + "&page=" + str(i) #+"chocolate%20cake&sort=re"
			print(dish, url)
			total = scrap(dish, f, url, total)
	f.close()

if __name__ == "__main__":
	main(3)