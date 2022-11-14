import logging
import os

import pandas as pd
import requests

import config


def load_IMDb_dataframe(file_name: str) -> pd.DataFrame:
    """This function loads the IMDb data into a pandas dataframe.
    The data is expected to be in the data/IMDb folder.

    Args:
        file_name (str): The name of the file to load.

    Returns:
        pandas.DataFrame: The data loaded into a pandas dataframe.
    """

    # get the data folder path, making it possible to run this script from anywhere
    data_path = os.path.join('..', 'data')

    # get the IMDb folder path
    imdb_path = os.path.join(data_path, 'IMDb')

    # load the data into a pandas dataframe
    df = pd.read_csv(os.path.join(imdb_path, file_name),
                     sep='\t', compression='gzip')

    return df


def find_in_moviedb(imdb_id: str, api_key: str = config.MOVIE_DB_API_KEY) -> dict:
    """This function finds the element with the given IMDb ID in the MovieDB database.

    Args:
        imdb_id (str): The IMDb ID of the element to find.
        api_key (str, optional): The MovieDB API key. Defaults to config.MOVIE_DB_API_KEY.

    Returns:
        dict: The element found in the MovieDB database. If no element is found, it returns an empty dictionary.
    """

    def _flatten(list_to_flatten: list) -> list:
        return [item for sublist in list_to_flatten for item in sublist]

    # check if the IMDb ID is valid
    if not imdb_id.startswith('tt') and len(imdb_id) != 9 and not imdb_id[2:].isdigit():
        logging.warning('Invalid IMDb ID: %s', imdb_id)
        return {}

    # make the request
    response = requests.get(f'https://api.themoviedb.org/3/find/{imdb_id}',
                            params={'api_key': api_key, 'external_source': 'imdb_id'})

    # check if the request was successful
    if response.status_code != 200:
        logging.warning('Error %d: %s', response.status_code, response.text)
        return {}

    # transform response into a list
    response_list = _flatten(list(response.json().values()))

    # if there is more than one element, it's an unexpected error
    if len(response_list) > 1:
        error_message = f'Unexpected error: more than one element found for IMDb ID {imdb_id}'
        logging.warning(error_message)
        raise ValueError(error_message)

    return response_list[0] if response_list else {}


if __name__ == '__main__':
    # df = load_IMDb_dataframe('title.basics.tsv.gz')
    print(find_in_moviedb('tt0000001'))
