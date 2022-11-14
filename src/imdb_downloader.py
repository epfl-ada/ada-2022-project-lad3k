import os
import re

import requests
from bs4 import BeautifulSoup


def download_IMDb_data(force_update=False):
    """This function downloads the IMDb data from the IMDb website and saves it to the data folder.
    If the data is already downloaded, it will not download it again unless force_update is True.

    Args:
        force_update (bool, optional): If set to True, it will download the data even if it is already downloaded.
            Defaults to False.
    """

    # Download all the data from IMDb (https://datasets.imdbws.com/)
    # and save it to data/IMDb

    # get the page
    response = requests.get('https://datasets.imdbws.com/')
    soup = BeautifulSoup(response.text, 'html.parser')

    # list all the links
    links = soup.find_all('a', href=re.compile(
        r'^https://datasets.imdbws.com/.*\.gz$'))

    # get the data folder path, making it possible to run this script from anywhere
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data')

    # create the IMDb folder if it doesn't exist
    imdb_path = os.path.join(data_path, 'IMDb')
    if not os.path.exists(imdb_path):
        os.makedirs(imdb_path)

    # download all the files
    for link in links:
        file_name = link['href'].split('/')[-1]

        # safe path
        file_path = os.path.join(data_path, 'IMDb', file_name)

        if not os.path.exists(file_path) or force_update:
            print('Downloading', file_name)
            response = requests.get(link['href'])
            with open(file_path, 'wb') as f:
                f.write(response.content)
        else:
            print('Skipping', file_name)


if __name__ == '__main__':
    download_IMDb_data()
