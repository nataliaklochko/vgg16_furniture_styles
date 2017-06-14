# -- coding: utf-8 --


from bs4 import BeautifulSoup as bs
from tqdm import tqdm
import urllib.request 
from PIL import Image
import requests
from io import BytesIO
import os

data_set_dir = 'data_set'
#dir_name_test = 'test_set'

def main():
	styles = ['mid-century-modern', 'modern', 'art-deco',
			  'scandinavian-modern', 'neoclassical', 'victorian',
			  'georgian', 'hollywood-regency', 'louis-xvi',
			  'art-nouveau', 'louis-xv', 'regency',
			  'industrial', 'baroque', 'folk-art',
			  'empire', 'rococo', 'arts-and-crafts',
			  'american-modern']

	try:
		os.stat(data_set_dir)
	except:
		os.mkdir(data_set_dir)

	for style in tqdm(styles):
		get_images(style)

def get_images(style):

	for i in range(1,101):
		try:
			url = 'https://www.1stdibs.com/furniture/style/' + style
			url += '/?page='+ str(i) + '&content=collapsed'
			html_doc = urllib.request.urlopen(url).read()
			soup = bs(html_doc, 'html.parser')


			for link in soup.select("img[src^=http]"):
				lnk = link["src"]
				response = requests.get(lnk)
				img = Image.open(BytesIO(response.content))
				img_name = style + '_' + str(lnk.split('/')[-1])
				if img.size[0] > 200 and img.size[1] > 200:
					img.save(os.path.join(data_set_dir, img_name))

		except Exception:
			pass

if __name__ == '__main__':
	main()
