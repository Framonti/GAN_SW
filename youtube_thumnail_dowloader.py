# created by Hritwik Singhal on 28-05-2018: 03:20
# https://github.com/HritwikSinghal/Youtube-Thumbnail-Downloader
# Modified by Francesco Monti on 27/01/2020

import os
import re
import requests
import json

from PIL import Image
from bs4 import BeautifulSoup
from config import PROJECT_ABSOLUTE_PATH


def download_image(ID, url, title=None):
    # insert ID in url of image
    img_url = 'https://img.youtube.com/vi/' + ID + '/maxresdefault.jpg'

    # Naming of images
    if title is None:
        res = requests.get(url)
        soup = BeautifulSoup(res.text, 'html.parser')
        title = str(soup.title)
        title = re.findall(r'<title>(.*)- YouTube</title>', title)[0].strip()

    name = f'{title}.jpg'

    # downloading the image
    print(f'Downloading "{title}".....')
    raw_data = requests.get(img_url, stream=True)
    save_path = os.path.join(os.path.join(PROJECT_ABSOLUTE_PATH, 'yt_thumbnail'), name)
    with open(save_path, "wb") as raw_img:
        for chunk in raw_data.iter_content(chunk_size=2048):
            if chunk:
                raw_img.write(chunk)
    print("Download Successful")

    img = Image.open(save_path)

    # resizing the image
    # og size: 1280x720
    with open(os.path.join(PROJECT_ABSOLUTE_PATH, 'params.json'), 'r') as params_file:
        params = json.load(params_file)
    resize_width = params['image_width']
    resize_height = params['image_height']
    img = img.resize((resize_width, resize_height))

    img.save(save_path)


def main():
    load_path = os.path.join(PROJECT_ABSOLUTE_PATH, 'yt_urls.txt')
    with open(load_path, 'r') as urls_file:
        i = 0
        for line in urls_file.read().split("\n"):
            if line.strip().startswith('https://www.youtube.com/'):
                url = line
                video_id = url.strip()[32:43]
                try:
                    download_image(video_id, url, i)
                    i += 1
                except:
                    raise ValueError(f'There was some errors in the URLs provided: {url}')


if __name__ == '__main__':
    main()
