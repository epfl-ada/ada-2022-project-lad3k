# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3.9.13 ('ada_project')
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import json


# %% [markdown]
# # CMU Dataset exploration

# %%
# load the CMU movies dataset
df = pd.read_csv('data/CMU/movie.metadata.tsv.gz',
                 sep='\t', header=None, compression='gzip')

# %%
df.columns = [
    'wikipedia_id',
    'freebase_id',
    'movie_name',
    'release_date',
    'boxoffice_revenue',
    'runtime',
    'languages',
    'countries',
    'genres'
]

# %%
df.release_date = pd.to_datetime(df.release_date, errors='coerce')
df.release_date.describe(datetime_is_numeric=True)

# %%
df.release_date.apply(lambda x: x.year).plot(kind='hist', bins=100)

# %% [markdown]
# ### Interpretation
# Looking at the statistics and the plot
# of the release dates of the CMU dataset,
# we can see that it mostly contains
# old movies, only 1/4 of the movies have been released
# after 2004, thus most movies are old,
# this means that if we use this dataset to study
# which streaming service is best our
# dataset will have a bias towards older movies.

# %%
df.languages = df.languages.apply(json.loads)
df.languages = df.languages.apply(lambda x: tuple(list(x.values())))
# filter movies with more than one language
df = df[df.languages.apply(lambda x: len(x) == 1)]
df.languages = df.languages.apply(lambda x: x[0])
# plot number of movies in each language
df.languages.groupby(df.languages).count().sort_values(
    ascending=False).head(20).plot(kind='bar')

# %%
# number of english movies
df[df.languages == 'English Language'].shape[0]

# %%
# number of non english movies
df[df.languages != 'English Language'].shape[0]

# %% [markdown]
# ### Interpretation
# The dataset contains about 50% english movies amd 50% movies in a different language than english,
# looking at movies that only have one spoken language in them. This shows a bias towards english movies

# %% [markdown]
# ### Further investigation of the CMU dataset to see if it is suitable to our problem

# %% [markdown]
# On the CMU website we see that the movies have been collected from the Noverber 4,
# 2012 dump of Freebase which in part explains why most of the movies are old,
#  if we get data from google trends on the terms streaming services
# and a specific streaming service for example Netflix,
# we can see that the streaming services industry started booming after most of the movies
# in the CMU dataset have been released

# %% [markdown]
# ![images/streaming.png](images/streaming.png)

# %% [markdown]
# ![images/netflix.png](images/netflix.png)

# %% [markdown]
# ![images/originals.png](images/originals.png)

# %% [markdown]
# This argument is not a rigourous one but more of an emperical argument in favor of not using the CMU dataset.
# For the reasons mentioned we will not use the CMU dataset to study which streaming service is best, but we will
# construct our own dataset using the imdb dataset and the moviedb api.

# %% [markdown]
# # Descripton of the newly constructed dataset

# %% [markdown]
# ### IMDb dataset

# %% [markdown]
# 1) Lets look at the movies present in  IMDBb

# %%
df_movies = pd.read_csv('data/IMDb/title.basics.tsv.gz',
                        sep='\t', compression='gzip')
df_movies.head()

# %%
# Only keep movies
df_movies = df_movies[df_movies.titleType == 'movie']
nb_movies = len(df_movies)
print(nb_movies)

# %%
nb_with_no_rdate = len(df_movies[df_movies['startYear'] == '\\N'])
print(nb_with_no_rdate)

# %%
# percentage of movies with no release date
nb_with_no_rdate / nb_movies * 100

# %%
# plot the number of movies per year in a histogram
df_movies['startYear'] = pd.to_numeric(df_movies['startYear'], errors='coerce')
df_movies['startYear'].plot(kind='hist', bins=100)

# %%
df_movies['startYear'].describe()

# %% [markdown]
# The dataset contains more recent movies the median is at the year :
# 2003 instead of 1985 (CMU dataset)

# %% [markdown]
# ### The scraped data from the moviedb api

# %%
df_new = pd.read_csv('data/moviedb_data.tsv.gz', sep='\t', compression='gzip')

# %%
df_new.columns

# %%
df_new.head(5)

# %%
len(df_new)

# %% [markdown]
# Given the imdb_id we scrape the overview, providers, bugdet, revenue, production
# company, production country provided by the moviedb api we have a total of 289093 movies, obtained using the imdb_id

# %%
set(map(type, df_new['providers']))

# %%
# Remove movies that dont have providers
df_new = df_new[df_new['providers'] != '{}']

# %%
# Parse the movies into python dicts
df_new['providers'] = df_new['providers'].apply(lambda x: json.dumps(x))

# %%
# Remove movies that dont have US streaming providers
df_new = df_new[df_new['providers'].apply(
    lambda prov_dict: ('US' in prov_dict))]


# %%
# Parses the providers into dicts
def get_json(row):
    return json.loads(row['providers'].replace("'", '"')[1:-1])


# %%
df_new['providers'] = df_new.apply(get_json, axis=1)

# %%
# Get the us providers list
df_new['providers'] = df_new['providers'].apply(lambda x: x['US'])

# %%
# Keep only movies that have a non empty list of providers
df_new = df_new[df_new['providers'].apply(lambda x: len(x) > 0)]

# %%
len(df_new)

# %% [markdown]
# We have a list of about 40 000 movies to use for our analysis,
# if we replace the "US" country code with "CH" we only get about
# 20 000 movies thus we decided to focus on the US market.

# %%
df_new['providers'] = df_new['providers'].apply(lambda x: tuple(x))

# %%
test = list(set(df_new['providers']))

# %%
# flatten the list of tuples
test = [item for sublist in test for item in sublist]

# %%
set(test)

# %%
len(set(test))

# %% [markdown]
# We have a list of 134 providers, we will keep all of them for now
#  and see if will reduce the number of providers later on.
# As an example we have the streaming service "

# %%
df_new.groupby('overview').count().sort_values(
    'imdb_id', ascending=False).head(20)

# %%
df_new

# %%
df_joined = df_movies.merge(
    left_on='tconst', right_on='imdb_id', right=df_new, how='inner')

# %%
df_joined.columns

# %%
df_joined.isAdult = pd.to_numeric(df_joined.isAdult)

# %%
set(df_joined.endYear)

# %%
# remove column from dataframe
df_joined = df_joined.drop(['endYear', 'tconst', 'titleType'], axis=1)

# %%
df_joined = df_joined.drop('originalTitle', axis=1)

# %%
df_crew = pd.read_csv('data/IMDb/title.crew.tsv.gz',
                      sep='\t', compression='gzip')
df_names = pd.read_csv('data/IMDb/name.basics.tsv.gz',
                       sep='\t', compression='gzip')

# %%
df_joined = df_crew.merge(
    left_on='tconst', right_on='imdb_id', right=df_joined, how='inner')

# %%
df_joined = df_joined.drop('tconst', axis=1)

# %% [markdown]
# We now have the director and writers for each movie in our dataset,
# they are identified by their imdb_id, we can lookup
# the information about them in the df_names dataframe

# %%
df_joined

# %% [markdown]
# We will now add columns that correspond to ratings in our dataframe

# %%
df_ratings = pd.read_csv('data/IMDb/title.ratings.tsv.gz',
                         sep='\t', compression='gzip')

# %%
df_joined = df_ratings.merge(
    left_on='tconst', right_on='imdb_id', right=df_joined, how='inner')

# %%
df_joined = df_joined.drop('tconst', axis=1)

# %%
df_joined.columns

# %%
df_joined.set_index('imdb_id', inplace=True)

# %%
df_joined.head(5)

# %%
df_joined.columns

# %% [markdown]
# Lets explore what we have in the newly constructed dataset,
#  and check if it is suitable for our problem

# %%
df_joined = df_joined.rename(columns={'startYear': 'release_year'})

# %%
# plot the release year of the movies
df_joined['release_year'].plot(kind='hist', bins=100)

# %%
df_joined['release_year'].describe()

# %%
# check for null values in columns
for col in df_joined.columns.to_list():
    print(col, df_joined[col].isnull().sum())

# %%
# filter the 5 movies that have no release year
df_joined[df_joined['release_year'].isnull()]

# %%
# update the missing values by looking up the values on google
df_joined.at['tt0417131', 'release_year'] = 1983.0
df_joined.at['tt12325302', 'release_year'] = 2019.0
df_joined.at['tt12325326', 'release_year'] = 2020.0
df_joined.at['tt14772866', 'release_year'] = 2022.0
df_joined.at['tt7971674', 'release_year'] = 2020.0


# %%
# check for missing values in the imdb columns identified by string '\\N'
for col in df_joined.columns.to_list():
    print(col, len(df_joined[df_joined[col].apply(lambda x: x == '\\N')]))

# %% [markdown]
# We have less than 10% of the movies that have missing values,
# we will see how we treat missing values, some possible solutions
# are to drop the rows with missing values, or to impute the missing values.
# The overview column always have a value but this does not mean that it is clean.

# %%
df_joined.groupby('overview').count().sort_values(
    ascending=False, by='averageRating').head(10)

# %% [markdown]
# Some overview values indicate that the value is missing like "No Overview" or
# "No overview found." and some overviews are quite general like "Bollywood 1972"
# or "Mexican feature film" and some are comletely wrong like
# "No one has entered a biography for him." or "What the movie has in store for you,
#  wait and watch this space for more updates.". We will have to find a way to keep only
# rows that have a correct overview, a good start would be to keep only the ones that are unique
# and have a length greater than 10.

# %%
# look at the distribution of the average rating
df_joined['averageRating'].plot(kind='hist', bins=100)


# %%
df_joined['averageRating'].describe()

# %%
# plot the runtime of the movies
df_joined['runtimeMinutes'] = pd.to_numeric(
    df_joined['runtimeMinutes'], errors='coerce')
df_joined['runtimeMinutes'].plot(kind='hist', bins=100, xlim=(50, 200))

# %%
df_joined['runtimeMinutes'].describe()

# %%
df_joined.query('runtimeMinutes == 720')

# %%
df_joined['providers']

# %% [markdown]
# Lets focus on the most popular streaming services in the US,
# to check the
# number of movies available in each streaming service namely:
# - Netflix
# - Hulu
# - Amazon Prime Video
# - Disney+
# - HBO Max
# - Peacock
# - Apple TV+

# %%
df_joined['on_netflix'] = df_joined['providers'].apply(lambda x: (
    'Netflix' or 'Netflix Kids' or 'Netflix basic with Ads') in x)
df_joined['on_prime'] = df_joined['providers'].apply(
    lambda x: 'Amazon Prime Video' in x)
df_joined['on_apple'] = df_joined['providers'].apply(
    lambda x: 'Apple TV Plus' in x)
df_joined['on_hulu'] = df_joined['providers'].apply(lambda x: 'Hulu' in x)
df_joined['on_disney'] = df_joined['providers'].apply(
    lambda x: 'Disney Plus' in x)
df_joined['on_hbo'] = df_joined['providers'].apply(lambda x: 'HBO Max' in x)
df_joined['on_peacock'] = df_joined['providers'].apply(
    lambda x: ('Peacock' or 'Peacock Premium') in x)


# %%
# keep only rows that have exactly one true in one of the streaming service columns and all others are false
df_joined = df_joined[(df_joined[['on_netflix', 'on_prime', 'on_apple',
                       'on_hulu', 'on_disney', 'on_hbo', 'on_peacock']].sum(axis=1) == 1)]
nb_per_streaming_serv = df_joined[[
    'on_netflix', 'on_prime', 'on_apple', 'on_hulu', 'on_disney', 'on_hbo', 'on_peacock']].sum()
print(f'Total number of movies: {nb_per_streaming_serv.sum()}')
nb_per_streaming_serv

# %%
nb_per_streaming_serv.sort_values(ascending=False).plot(kind='bar')

# %% [markdown]
# The data is not uniformlly distributed, some streaming services have a
# lot of movies and some have very few movies, we will have to find a way
# to deal with this problem when performing our analysis.

# %%
df_joined['genres'] = df_joined['genres'].apply(lambda x: x.split(','))

# %%
genres = set(df_joined['genres'].sum())

# %%
genres = list(genres)

# %%
movies_genre = {}
for genre in genres:
    movies_genre[genre] = df_joined['genres'].apply(lambda x: genre in x).sum()

# %%
df_movies_genre = pd.DataFrame.from_dict(
    movies_genre, orient='index', columns=['nb_movies'])

# %%
# categories of movies on the steaming services defiened above
df_movies_genre.sort_values(by='nb_movies', ascending=False).plot(kind='bar')

# %% [markdown]
# Movies can have multiple categories, for example one movie might have the category action and drama.
