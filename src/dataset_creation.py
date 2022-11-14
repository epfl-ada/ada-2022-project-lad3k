import logging
import os

import pandas as pd
import requests
from tqdm import tqdm

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
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data')

    # get the IMDb folder path
    imdb_path = os.path.join(data_path, 'IMDb')

    # load the data into a pandas dataframe
    df = pd.read_csv(os.path.join(imdb_path, file_name),
                     sep='\t', compression='gzip')

    return df


def find_in_moviedb(imdb_id: str, api_key: str = config.MOVIE_DB_API_KEY) -> int:
    """This function finds the element with the given IMDb ID in the MovieDB database.

    Args:
        imdb_id (str): The IMDb ID of the element to find.
        api_key (str, optional): The MovieDB API key. Defaults to config.MOVIE_DB_API_KEY.

    Returns:
        id (int): The ID of the element with the given IMDb ID in the MovieDB database. -1 if not found.
    """

    def _flatten(list_to_flatten: list) -> list:
        return [item for sublist in list_to_flatten for item in sublist]

    # check if the IMDb ID is valid
    if not imdb_id.startswith('tt') and len(imdb_id) != 9 and not imdb_id[2:].isdigit():
        logging.warning('Invalid IMDb ID: %s', imdb_id)
        return -1

    # make the request
    response = requests.get(f'https://api.themoviedb.org/3/find/{imdb_id}',
                            params={'api_key': api_key, 'external_source': 'imdb_id'})

    # check if the request was successful
    if response.status_code != 200:
        logging.warning('Error %d: %s', response.status_code, response.text)
        return -1

    # transform response into a list
    response_list = _flatten(list(response.json().values()))

    # if there is more than one element, it's an unexpected error
    if len(response_list) > 1:
        error_message = f'Unexpected error: more than one element found for IMDb ID {imdb_id}'
        logging.warning(error_message)
        raise ValueError(error_message)

    return response_list[0]['id'] if len(response_list) == 1 else -1


def get_movie_features(movie_id: int, api_key: str = config.MOVIE_DB_API_KEY) -> dict:
    """This function gets the features of the movie with the given ID from the MovieDB database.

    Args:
        movie_id (int): The ID of the movie to get the features from.
        api_key (str, optional): The MovieDB API key. Defaults to config.MOVIE_DB_API_KEY.

    Returns:
        dict: The features of the movie with the given ID from the MovieDB database.
            The dict contains the following keys (all optional):
            - id: The IMDb ID of the movie.
            - providers: The streaming providers of the movie. {<country>: [<providers>]}
            - budget: The budget of the movie.
            - revenue: The revenue of the movie.
            - production_companies: The production companies of the movie.
            - production_countries: The production countries of the movie.
    """

    # make the request
    response = requests.get(f'https://api.themoviedb.org/3/movie/{movie_id}',
                            params={'api_key': api_key})

    # check if the request was successful
    if response.status_code != 200:
        logging.warning('Error %d (/movie): %s',
                        response.status_code, response.text)
        return {}

    # get the response as a dict
    movie_response = response.json()

    # get the streaming providers
    response = requests.get(f'https://api.themoviedb.org/3/movie/{movie_id}/watch/providers',
                            params={'api_key': api_key})

    # check if the request was successful
    if response.status_code != 200:
        logging.warning('Error %d (/movie/id/watch/providers): %s',
                        response.status_code, response.text)
        return {}

    # get the response as a dict
    providers_response = response.json()

    # get the streaming providers as a dict {<country>: [<providers>]}
    providers = {
        country: [provider['provider_name']
                  for provider in providers['flatrate']] if 'flatrate' in providers else []
        for country, providers in providers_response['results'].items()}

    # get the features
    features = {
        'imdb_id': movie_response['imdb_id'],
        'providers': providers,
        'budget': movie_response['budget'],
        'revenue': movie_response['revenue'],
        'production_companies': [company['name'] for company in movie_response['production_companies']],
        'production_countries': [country['iso_3166_1'] for country in movie_response['production_countries']]
    }

    return features


def create_moviedb_dataset(filename: str = 'moviedb_data.csv'):
    """This function creates the MovieDB dataset. It loads the IMDb data and finds the corresponding
    elements in the MovieDB database. Then, it gets the features of the movies and saves them in a
    CSV file.

    Args:
        filename (str, optional): The name of the file to save the dataset to. Defaults to 'moviedb_data.csv'.
    """

    # set logging level
    logging.basicConfig(level=logging.INFO)

    logging.info('Loading IMDb data...')
    df = load_IMDb_dataframe('title.basics.tsv.gz')

    # keep only unique IMDb IDs and store them in a list
    imdb_ids = df['tconst'].unique().tolist()

    # Create a new dataframe with the features of the movies
    movies_df = pd.DataFrame(
        columns=['imdb_id', 'providers', 'budget', 'revenue', 'production_companies', 'production_countries'])

    logging.info('Getting features from MovieDB...')
    for imdb_id in tqdm(imdb_ids):
        # find the movie in the MovieDB database
        movie_id = find_in_moviedb(imdb_id)

        # if the movie is not found, skip it
        if movie_id == -1:
            logging.warning(
                'Movie with IMDb ID %s not found in the MovieDB database', imdb_id)
            continue

        # get the features of the movie
        features = get_movie_features(movie_id)

        # if the features are not found, skip it
        if not features:
            logging.warning(
                'Features of movie with IMDb ID %s not found in the MovieDB database', imdb_id)
            continue

        # add the features to the dataframe with pandas.concat since append will is deprecated
        movies_df = pd.concat(
            [movies_df, pd.DataFrame([features])], ignore_index=True)

    logging.info('Saving features to CSV...')
    # save the dataframe to a CSV file
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
    movies_df.to_csv(os.path.join(data_path, filename), index=False)


if __name__ == '__main__':
    create_moviedb_dataset()
