import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def create_directory(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def download_image(url, folder_path):
    response = requests.get(url)
    if response.status_code == 200:
        file_name = os.path.join(folder_path, url.split("/")[-1])
        with open(file_name, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded: {file_name}")
    else:
        print(f"Failed to download: {url}")

def scrape_images(url, folder_path):

    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        img_tags = soup.find_all('img') 
        
        create_directory(folder_path)
        for img in img_tags:
            img_url = img.get('src')
            if img_url:
                img_url = urljoin(url, img_url)
                download_image(img_url, folder_path)
    else:
        print(f"Failed to retrieve the webpage: {url}")

target_url = "http://www.people.com.cn"
save_folder = "downloaded_images"
scrape_images(target_url, save_folder)