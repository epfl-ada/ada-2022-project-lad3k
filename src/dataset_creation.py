import logging
import os
import queue
import time
from multiprocessing import Pool
from threading import Thread

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
        logging.error(error_message)
        return -1

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
            - overview: The overview of the movie.
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
        'overview': movie_response['overview'],
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

    logging.info('Loading IMDb data...')
    df = load_IMDb_dataframe('title.basics.tsv.gz')

    df = df[df['titleType'] == 'movie']

    # keep only unique IMDb IDs and store them in a list
    imdb_ids = df['tconst'].unique().tolist()

    # sort the list
    imdb_ids.sort()

    # If the csv already exists, get the last IMDb ID that was processed
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
    csv_path = os.path.join(data_path, filename)
    csv_already_exists = os.path.exists(csv_path)

    if csv_already_exists:
        df = pd.read_csv(csv_path)
        last_imdb_id = df['imdb_id'].iloc[-1]
        last_imdb_id_index = imdb_ids.index(last_imdb_id)
        imdb_ids = imdb_ids[last_imdb_id_index + 1:]

    # We save the data in a CSV file each X movies
    # This is to avoid losing all the data if the program crashes
    for i in range(0, len(imdb_ids), 10000):
        upper_bound = min(i + 10000, len(imdb_ids))
        # get the next x IMDb IDs
        imdb_ids_subset = imdb_ids[i:upper_bound]

        # f'Processing IMDb IDs {imdb_ids_subset[0]}-{imdb_ids_subset[-1]}, representing <number> % of the data...'
        logging.info(
            f'Processing IMDb IDs {imdb_ids_subset[0]}-{imdb_ids_subset[-1]}, ' +
            f'representing {round(upper_bound / len(imdb_ids) * 100)} % of the data...')

        movies_df = get_movies_features_for_list_imdb_ids(
            imdb_ids_subset, nb_workers=10)
        movies_df.sort_values(by='imdb_id', inplace=True)

        if csv_already_exists:
            movies_df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            movies_df.to_csv(csv_path, index=False)
            csv_already_exists = True

        logging.info(f'Done processing IMDb IDs {imdb_ids_subset[0]}-{imdb_ids_subset[-1]}, ' +
                     f'representing {upper_bound / len(imdb_ids) * 100} % of the data...')


def get_movies_features_for_list_imdb_ids(imdb_ids: list, nb_workers: int = 10) -> pd.DataFrame:
    """This function gets the features of the movies given by imdb_ids

    Args:
        imdb_ids (list): The list of IMDb IDs of the movies to get the features from.
        nb_workers (int, optional): The number of workers to use. Defaults to 10.

    Returns:
        pd.DataFrame: The dataframe containing the features of the movies.
    """

    class Worker(Thread):
        def __init__(self, request_queue, error_queue):
            Thread.__init__(self)
            self.queue = request_queue
            self.error_queue = error_queue
            self.results = pd.DataFrame(
                columns=['imdb_id', 'overview', 'providers', 'budget', 'revenue', 'production_companies',
                         'production_countries'])

        def run(self):
            # needed to put the errors in the error queue
            try:
                while True:
                    should_stop = False
                    # get the next IMDb IDs
                    next_ids = []
                    for _ in range(10):
                        next_id = self.queue.get()
                        if next_id is None:
                            should_stop = True
                            break
                        next_ids.append(next_id)

                    if not next_ids:
                        break
                    # Get the features with a multiprocessing pool
                    with Pool(processes=5) as pool:
                        # pool error handling
                        try:

                            movies_ids = pool.map(find_in_moviedb, next_ids)

                            # Remove the -1 elements
                            movies_ids = [x for x in movies_ids if x != -1]

                            # use starmap and handle errors
                            features_list = pool.starmap(
                                get_movie_features, zip(movies_ids))

                            # Remove the empty dicts
                            features_list = [x for x in features_list if x]
                        except Exception as e:
                            self.error_queue.put(e)
                            break

                    for features in features_list:
                        self.results = pd.concat(
                            [self.results, pd.DataFrame([features])], ignore_index=True)

                    if should_stop or self.error_queue.qsize() > 0:
                        break
            except Exception as e:
                self.error_queue.put(e)

    def listener(queue, error_queue):
        total_length = len(imdb_ids) + nb_workers

        pbar = tqdm(total=total_length)
        # set pbar to (len(imdb_ids) - queue.qsize()) to show the progress of the requests
        last_pbar_value = 0
        while True:
            queue_size = queue.qsize()
            pbar.update((total_length - queue_size) - last_pbar_value)
            last_pbar_value = total_length - queue_size

            if queue.qsize() == 0 or error_queue.qsize() > 0:
                break
            time.sleep(1)

    # Create queue and add addresses
    q = queue.Queue()
    errors = queue.Queue()

    for imdb_id in imdb_ids:
        q.put(imdb_id)

    # Workers keep working till they receive an empty string
    for _ in range(nb_workers):
        q.put(None)

    # add listener to queue to print progress with tqdm
    listener_thread = Thread(target=listener, args=(q, errors))
    listener_thread.start()

    # Create workers and add tot the queue
    workers = []
    for _ in range(nb_workers):
        worker = Worker(q, errors)
        worker.start()
        workers.append(worker)

    # Join workers to wait till they finished
    for worker in workers:
        worker.join()

    # terminate the listener
    listener_thread.join()

    if not errors.empty():
        error = errors.queue[0]

        # check if connection error otherwise it may not stop the program
        if isinstance(error, requests.exceptions.ConnectionError):
            print('IF YOU GET CONNECTION ERRORS, IT WILL CRASH (ON PURPOSE) A SULUTION IS TO USE LESS WORKERS/THREADS',
                  flush=True)
            raise Exception(error)
        else:
            raise error

    # Combine results from all workers
    results = pd.DataFrame(
        columns=['imdb_id', 'overview', 'providers', 'budget', 'revenue', 'production_companies',
                 'production_countries'])
    for worker in workers:
        results = pd.concat(
            [results, worker.results], ignore_index=True)

    return results


if __name__ == '__main__':
    # set logging level
    logging.basicConfig(level=logging.INFO)

    create_moviedb_dataset()
