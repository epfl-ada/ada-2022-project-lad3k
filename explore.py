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
df = pd.read_csv('data/MovieSummaries/movie.metadata.tsv',
                 sep='\t', header=None)

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
# Looking at the statistics and the plot of the release dates of the CMU dataset,
# we can see that it mostly contains old movies, only 1/4 of the movies have been released
# after 2004, thus most movies are old, this means that if we use this dataset to study
# which streaming service is best our dataset will have a bias towards older movies.

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
# company, production country provided by the moviedb api

# %%
set(map(type, df_new['providers']))

# %%
df_new[df_new['providers'] != '{}'].count()

# %%
df_new.groupby('overview').count().sort_values(
    'imdb_id', ascending=False).head(20)

# %%
