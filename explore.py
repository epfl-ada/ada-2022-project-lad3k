# -*- coding: utf-8 -*-
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
#     display_name: Python 3.9.15 ('ada_project')
#     language: python
#     name: python3
# ---

# %%
import matplotlib.pyplot as plt
import string
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from gensim.models import Phrases
from nltk.corpus import stopwords
from nltk import pos_tag
from src.nlp_helper import get_wordnet_pos, build_dictionnary_and_corpus, create_lda_model, get_topics, \
    get_topic_distribution
import numpy as np
import nltk
import pandas as pd
import json


# %% [markdown]
# # 1. CMU Dataset 

# %% [markdown]
# ## 1.1 Exploration

# %% [markdown]
# We want to explore a movies dataset that includes movies in different streaming services. First, we explore the CMU Dataset and if we find it interesting we'll just have to add new columns representing the presence or not of each movie in each streaming service.

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
# ## 2. CMU Dataset suitability with our project

# %% [markdown]
# On the CMU website we see that the movies have been collected from the November 4,
# 2012 dump of Freebase. if we get data from google trends (see below),
# we can see that the streaming services industry started booming after most of the movies
# in the CMU dataset have been released. 

# %% [markdown]
# ![images/streaming.png](images/streaming.png)

# %% [markdown]
# ![images/netflix.png](images/netflix.png)

# %% [markdown]
# ![images/originals.png](images/originals.png)

# %% [markdown]
# We beleive that streaming services have generally more recent movies than old ones. To verify this we
# will construct our own dataset using the imdb dataset and the moviedb api. This hypothesis will be proved 
# later (see next section)

# %% [markdown]
# # 2. Descripton of our newly constructed dataset

# %% [markdown]
# ### 2.1 IMDb dataset

# %% [markdown]
# 1) Lets look at the movies present in  IMDBb

# %%
df_movies = pd.read_csv('data/IMDb/title.basics.tsv.gz',
                        sep='\t', compression='gzip')
df_movies.head()

# %%
# Only keep movies (no TV shows, no shorts, etc.)
df_movies = df_movies[df_movies.titleType == 'movie']
nb_movies = len(df_movies)
print(nb_movies)

# %%
# Only keep movies with a known startYear
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
# ### 2. The scraped data from the moviedb api

# %% [markdown]
# Given the imdb_id we scrape the overview, providers, bugdet, revenue, production
# company, production country provided by the moviedb api. 

# %%
df_new = pd.read_csv('data/moviedb_data.tsv.gz', sep='\t', compression='gzip')

# %%
df_new.columns

# %%
df_new.head(5)

# %%
len(df_new)

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
# and see if will reduce the number of providers later on.

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

# %% [markdown]
# ## NLP exploration
# We will now see if we can perform topics extraction over the movie plots.

# %% [markdown]
# ### Data Loading and basic analysis
# %%
df = pd.read_csv('data/moviedb_data.tsv.gz', sep='\t', compression='gzip')
df_plots = df.copy()
# keep only the overview and providers columns as we don't use others for NLP
df_plots = df_plots[['overview', 'providers']]

# %% [markdown]
# #### Overview Analysis
#
# In this NLP exploration, we are mostly interested by the overview and providers fields.
# Let's see if some movies don't contain overviews.

# %%
# print the percentage of movies with no overview
nb_with_no_overview = len(df_plots[df_plots['overview'].isnull()])
print(round(nb_with_no_overview / len(df_plots)
      * 100, 1), '% of movies have a null overview')

# replace the missing values with an empty string
df_plots['overview'] = df_plots['overview'].fillna('')

df_plots.head(8)['overview']

# %% [markdown]
# Here we see that some movies (5,7) contain a non empty overview, but which indicate that there is no overview for this
# movie.
# We can replace them by empty overviews. However this replacement may not be exhaustive if some useless plots are not
# $\\$ "Know what this is about?" but something else.

# %%
n_no_overview_0 = round(
    len(df_plots[df_plots['overview'].str.contains('Know what this is about?')]))
n_no_overview_1 = round(
    len(df_plots[df_plots['overview'].str.contains('No Overview')]))
n_no_overview_2 = round(
    len(df_plots[df_plots['overview'].str.contains('No overview found')]))
total_no_overview = n_no_overview_0 + n_no_overview_1 + \
    n_no_overview_2 + nb_with_no_overview
print(round(total_no_overview / len(df_plots) * 100, 1),
      '% of movies have a no overview')

# %% [markdown]
# 12% of movies without overview is tolerable given that our dataset is large, but firstly our search is not exhaustive,
# and secondly we will $\\$ have to check that most of the movies on the streaming platforms have an overview to be
# able to apply NLP.

# %%
# we start by replacing the strings 'No Overview', 'No overview found' and 'Know what this is about?' by an empty string
df_plots['overview'] = df_plots['overview'].apply(lambda x: x.replace(
    'No Overview', '').replace('No overview found', '').replace('Know what this is about?', ''))
# we clean provider data
df_plots['providers'] = df_plots['providers'].fillna('{}')

# %% [markdown]
# According to previous analysis, we decided to use providers of Switzerland and US, we will now see if the movies
# provided in these $\\$ countries possess enough plots.

# %%
# US provider movies
df_plots_us = df_plots[df_plots['providers'].str.contains('US')]
# keep only movies where plots isn't empty
df_plots_us_overview = df_plots_us[df_plots_us['overview'] != '']

n_plot_us_overview = len(df_plots_us_overview)

plt.title('Pie chart of movies containing overview in the US')
# plot the number of movies with overview in the US
plt.pie([n_plot_us_overview, len(df_plots_us) - n_plot_us_overview], labels=['Overview', 'No Overview'],
        autopct='%1.1f%%')
plt.show()

# %% [markdown]
# Thus we can see that we can work on an NLP for topics analysis for the movies provided in US as almost all
# movies have plots.
#
# ### Plot preparation
#
# Now that we have seen we have enough movies plots in the US and CH regions, we can work on our topics analysis. $\\$
# To make a simple first exploration of topics analysis, we will simplify by merging movies from CH and US together $\\$
# as language is roughly the same. We could however try to split between the two in further analysis.
#
# We will transform the plots in order to make them intepretable by an LDA model. This includes
# - Tokenization
# - Lemmatization
# - Removing of stopwords
#
# This is usefull as we want to find ressemblance between words, so we should replace words with same meaning by one
# common word.
# We also want to remove most commun words. This allows to remove low-information words, allowing our
# model to focus on important $\\$ words.
#
# For the below part, we use a smaller dataset for performance reasons as we are trying nlp techniques,
#  but in the future we will use $\\$ the full dataset.

# %% [markdown]
# #### Tokenization

# %%
# take a sample of 30% of the movies
df_plots_us = df_plots_us_overview.sample(frac=0.1, random_state=42)
print(f'Number of movies in the sample: {len(df_plots_us)}')

# Tokenize the plots
df_plots_us['tokenized_plots'] = df_plots_us['overview'].apply(
    lambda movie_plot: word_tokenize(movie_plot))
df_plots_us.head()['tokenized_plots']

# %% [markdown]
# #### Lemmatization
# we start by assocating a POS tag to each word (i.e if a word is a Noun, Verb, Adjective, etc.)

# %%
df_plots_us['plots_with_POS_tag'] = df_plots_us['tokenized_plots'].apply(
    lambda tokenized_plot: pos_tag(tokenized_plot))
df_plots_us['plots_with_POS_tag'].head()

# %% [markdown]
# If a word has no tag we don't change it. However if there is a tag, we lemmatize the word according to its tag.

# %%
nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()
# Lemmatize each word given its POS tag
df_plots_us['lemmatized_plots'] = df_plots_us['plots_with_POS_tag'].apply(
    lambda tokenized_plot: [word[0] if get_wordnet_pos(word[1]) == ''
                            else lemmatizer.lemmatize(word[0], get_wordnet_pos(word[1])) for word in tokenized_plot])
df_plots_us['lemmatized_plots'].head()

# %% [markdown]
# #### Stop words removal

# %%
# list of stop words could be improved
stop_words = ['\'s']
all_stopwords = stopwords.words(
    'English') + list(string.punctuation) + stop_words

# %%

# remove the white space inside each words
df_plots_us['plots_without_stopwords'] = df_plots_us['lemmatized_plots'].apply(
    lambda tokenized_plot: [word.strip() for word in tokenized_plot])
# lowercase all words in each plot
df_plots_us['plots_without_stopwords'] = df_plots_us['plots_without_stopwords'].apply(
    lambda plot: [word.lower() for word in plot])
# remove stopwords from the plots
df_plots_us['plots_without_stopwords'] = df_plots_us['plots_without_stopwords'].apply(
    lambda plot: [word for word in plot if word not in all_stopwords])
# remove word if contains other letter than a-z or is a single character
df_plots_us['plots_without_stopwords'] = df_plots_us['plots_without_stopwords'].apply(
    lambda plot: [word for word in plot if word.isalpha() and len(word) > 1])
df_plots_us['plots_without_stopwords'].head()[0:2]

# %%
before_stop_words_total_number_of_words =\
    len([word for sentence in df_plots_us['lemmatized_plots']
        for word in sentence])
after_stop_words_total_number_of_words =\
    len([word for sentence in df_plots_us['plots_without_stopwords']
        for word in sentence])
print('We kept {}% of the words in the corpus'.format(
    round(after_stop_words_total_number_of_words/before_stop_words_total_number_of_words, 2) * 100))

# %% [markdown]
# ### Latent Direchlet Allocation
# We need to create a list of tokens, i.e words that will be used inside our dictionary (depending on their frequency).
# $\\$
# We can start by creating bi-gram for some words (represent to one words by one unique composed word)
# It can be also interesting to see $\\$ if creating tri-gram allows to extract more information from plots.

# %%
tokens = df_plots_us['plots_without_stopwords'].tolist()
bigram_model = Phrases(tokens)
tokens = list(bigram_model[tokens])
print(tokens[0:2])

# %% [markdown]
# #### Hyperparameters

# %%
no_below = 60  # minimum number of documents a word must be present in to be kept
no_above = 0.5  # maximum proportion of documents a word can be present in to be kept
n_topics = 10  # number of topics
n_passes = 10  # number of passes through the corpus during training

# %% [markdown]
# ### Dictionnary & Corpus
# The dictionnary will be the list of unique words, and the corpus a list of movie plots bag of words.

# %%
# we create a dictionary that maps each word to a unique integer
# we also create a corpus. Each movie plot is encoded as a bag of words in the corpus.
# A bag of word means that we count the number of times each word appears in the mvoie plot
dictionary, corpus = build_dictionnary_and_corpus(
    tokens, no_below=no_below, no_above=no_above)
print('Dictionary size: {}'.format(len(dictionary)))
print('Dictionary first 10 elements: {}'.format(
    list(dictionary.items())[0:10]))
print('Corpus size: {}'.format(len(corpus)))
print('Corpus first 2 elements: {}'.format(corpus[0:2]))

# %% [markdown]
# #### LDA Model

# %%
np.random.seed(9999)
lda_model = create_lda_model(
    corpus, dictionary, num_topics=n_topics, passes=n_passes)

# %%
# get the topics
topics = get_topics(lda_model, num_topics=n_topics, num_words=10)
# print topics with new line
for i, topic in enumerate(topics):
    print('Topic {}: {}'.format(i, topic))

# %% [markdown]
# From these topics, we could interpret the 0th as more drama movies, whereas the 2th could be more adventurous movies.
# However some others topics aren't really good, we cannot find a true ressemblance between the words inside.
# We will work in the future of this project on how to improve our model to have miningfull topics.

# %%
# for each movie plot, get its topic distribution (i.e the probability of each topic) in descending order
topic_distributions = get_topic_distribution(lda_model, corpus)
print('Movie plot: {}'.format(df_plots['overview'].iloc[0]))
print('Topic distribution for the first movie plot: {}'.format(
    topic_distributions[0]))

# %% [markdown]
# Here we see that the movie plot is mainly disrtributed on topics 7,1,0 which are related to love, traveling,
# and drama.
# This could actually make sens given the movie plot.
