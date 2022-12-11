import json
import pandas as pd


def prepare_df():
    df_movies = pd.read_csv('data/IMDb/title.basics.tsv.gz',
                            sep='\t', compression='gzip')
    df_movies['startYear'] = pd.to_numeric(
        df_movies['startYear'], errors='coerce')

    df_new = pd.read_csv('data/moviedb_data.tsv.gz',
                         sep='\t', compression='gzip')
    # Remove movies that dont have providers
    df_new = df_new[df_new['providers'] != '{}']
    # Parse the movies into python dicts
    df_new['providers'] = df_new['providers'].apply(lambda x: json.dumps(x))
    # Remove movies that dont have US streaming providers
    df_new = df_new[df_new['providers'].apply(
        lambda prov_dict: ('US' in prov_dict))]
    # Parses the providers into dicts

    def get_json(row):
        return json.loads(row['providers'].replace("'", '"')[1:-1])

    df_new['providers'] = df_new.apply(get_json, axis=1)
    # Get the US providers list
    df_new['providers'] = df_new['providers'].apply(lambda x: x['US'])
    # Keep only movies that have a non empty list of providers
    df_new = df_new[df_new['providers'].apply(lambda x: len(x) > 0)]

    df_new['providers'] = df_new['providers'].apply(lambda x: tuple(x))

    df_joined = df_movies.merge(
        left_on='tconst', right_on='imdb_id', right=df_new, how='inner')
    df_joined.isAdult = pd.to_numeric(df_joined.isAdult)

    df_joined = df_joined.drop(['endYear', 'tconst', 'titleType'], axis=1)
    df_joined = df_joined.drop('originalTitle', axis=1)
    df_crew = pd.read_csv('data/IMDb/title.crew.tsv.gz',
                          sep='\t', compression='gzip')
    # df_names = pd.read_csv('data/IMDb/name.basics.tsv.gz',
    #                   sep='\t', compression='gzip')
    df_joined = df_crew.merge(
        left_on='tconst', right_on='imdb_id', right=df_joined, how='inner')
    df_joined = df_joined.drop('tconst', axis=1)

    df_ratings = pd.read_csv('data/IMDb/title.ratings.tsv.gz',
                             sep='\t', compression='gzip')
    df_joined = df_ratings.merge(
        left_on='tconst', right_on='imdb_id', right=df_joined, how='inner')
    df_joined = df_joined.drop('tconst', axis=1)

    df_joined.set_index('imdb_id', inplace=True)

    df_joined = df_joined.rename(columns={'startYear': 'release_year'})

    df_joined.at['tt0417131', 'release_year'] = 1983.0
    df_joined.at['tt12325302', 'release_year'] = 2019.0
    df_joined.at['tt12325326', 'release_year'] = 2020.0
    df_joined.at['tt14772866', 'release_year'] = 2022.0
    df_joined.at['tt7971674', 'release_year'] = 2020.0
    df_joined['runtimeMinutes'] = pd.to_numeric(
        df_joined['runtimeMinutes'], errors='coerce')

    df_joined['on_netflix'] = df_joined['providers'].apply(lambda x: (
        'Netflix' or 'Netflix Kids' or 'Netflix basic with Ads') in x)
    df_joined['on_prime'] = df_joined['providers'].apply(
        lambda x: 'Amazon Prime Video' in x)
    # keep only movies that are on prime or on netflix
    df_joined = df_joined.query('on_netflix == True or on_prime == True')
    return df_joined
