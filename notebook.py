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
#     display_name: ada_project
#     language: python
#     name: python3
# ---
# %%
import string

import matplotlib.pyplot as plt
import networkx as nx
import nltk
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pycountry
import seaborn as sns
from gensim.models import Phrases
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from PIL import Image
from plotly.subplots import make_subplots
from scipy.stats import bootstrap
from scipy.stats import ttest_ind
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from textblob import TextBlob
from tqdm import tqdm
from wordcloud import WordCloud

from src import helper
from src.helper import prepare_df
from src.nlp_helper import build_dictionnary_and_corpus
from src.nlp_helper import create_lda_model
from src.nlp_helper import get_topic_distribution
from src.nlp_helper import get_topics
from src.nlp_helper import get_wordnet_pos

# Constants
NETFLIX_COLOR = '#E50914'
PRIME_COLOR = '#00A8E1'

# fix a seed for reproducibility
np.random.seed(42)
# %% [markdown]
# ---
# # Data Exploration

# %%
df = prepare_df()
df.head()

# %% [markdown]
# As of April 2022, there are 4901 movies on netflix, we have 2915 movies in the dataset thus we can estimate
# that we are capturing about 60% of the movies on the streaming service.
# For Prime there are 6985 movies and we have 6981 so we are capturing about 99.9%.
#
# *[source](https://blog.reelgood.com/which-streaming-service-offers-the-best-bang-for-your-buck)*

# %% [markdown]
# ![movies](images/movies_ss.png)

# %% [markdown]
# Can also check for quality movies and high quality movies the fraction of movies that are in the dataset.

# %%
# create a new column called platform with Netflix if only 'on_netflix' is True and Prime if only 'on_prime' is True
#  else 'both'
df['platform'] = df.apply(lambda x: 'Netflix' if x['on_netflix'] and not x['on_prime']
                          else 'Prime' if x['on_prime'] and not x['on_netflix'] else 'Both', axis=1)

# %%
# plot the number of movies on netflix and on prime and on both using the 'platform' column
fig = px.histogram(df, x='platform', color='platform', color_discrete_map={
                   'Netflix': NETFLIX_COLOR, 'Prime': PRIME_COLOR, 'Both': 'grey'})
# shade the color based on the percentage of ratings in the 'averageRating' column
fig.update_traces(marker_line_color='black', marker_line_width=0, opacity=0.8)
fig.update_layout(title='Number of movies per platfrom in our dataset',
                  xaxis_title='Platform', yaxis_title='Number of movies')


fig.show()
# generate html of plotly figure
fig.write_html('html/number_of_movies_per_platform.html')

# %% [markdown]
# ### Movies on Netflix and Prime (US)
# - Plot the genre distribution of movies on Netflix and Prime
# - Plot the production companies
# - Plot the production countries
# - Plot the directors and wirters
# - Plot the release year
# - Plot the runtime
# - Plot the average rating
#


# %%
df['streaming_service'] = ['both' if netflix and prime else 'netflix' if netflix
                           else 'prime' for netflix, prime in zip(df['on_netflix'], df['on_prime'])]

# %%
df.shape

# %%
# get dataframe with streaminung service = both
df_both = df[df['streaming_service'] == 'both']
df_both.shape

# %% [markdown]
# >It's very rare to find a movie that's on both platforms

# %%
# get movies on netflix and prime
netflix_movies = df[df['on_netflix']]
prime_movies = df[df['on_prime']]


# %%
# get unique genres function
def get_unique_genres(df):
    genres = set(df['genres'].apply(lambda x: x.split(',')).sum())
    genres = list(genres)
    return genres


# %%
netflix_genres = get_unique_genres(netflix_movies)
movies_genre = {}
for genre in netflix_genres:
    movies_genre[genre] = netflix_movies['genres'].apply(
        lambda x: genre in x).sum()
netflix_genre = pd.DataFrame.from_dict(
    movies_genre, orient='index', columns=['nb_movies'])

prime_genres = get_unique_genres(prime_movies)
movies_genre = {}
for genre in prime_genres:
    movies_genre[genre] = prime_movies['genres'].apply(
        lambda x: genre in x).sum()
prime_genre = pd.DataFrame.from_dict(
    movies_genre, orient='index', columns=['nb_movies'])

# join the two dataframes without losing the genres
both_genre = netflix_genre.join(
    prime_genre, how='outer', lsuffix='_netflix', rsuffix='_prime')
both_genre = both_genre.fillna(0)

# create a new column with the normalized sum of movies on netflix and prime
both_genre['total_movies'] = (both_genre['nb_movies_netflix'] + both_genre['nb_movies_prime']) / (both_genre[
    'nb_movies_netflix'].sum() + both_genre['nb_movies_prime'].sum())

# normalize the number of movies
both_genre['nb_movies_netflix'] = both_genre['nb_movies_netflix'] / \
    both_genre['nb_movies_netflix'].sum()
both_genre['nb_movies_prime'] = both_genre['nb_movies_prime'] / \
    both_genre['nb_movies_prime'].sum()


both_genre = both_genre.sort_values(by='total_movies', ascending=False)
both_genre.drop('total_movies', axis=1, inplace=True)
both_genre.plot(kind='bar', figsize=(15, 5),
                color=[NETFLIX_COLOR, PRIME_COLOR])


plt.title('Movies genre on Netflix and Prime')
plt.xlabel('Genre')
plt.ylabel('Normalized Frequency')
plt.show()


# %%
netflix_movies

# %%

both_genre = both_genre[:10]
both_genre = both_genre.reset_index()


fig = go.Figure()

fig.add_trace(go.Scatterpolar(
    r=both_genre['nb_movies_netflix'],
    theta=both_genre['index'],
    fill='toself',
    name='Netflix',
    marker_color=NETFLIX_COLOR

))
fig.add_trace(go.Scatterpolar(
    r=both_genre['nb_movies_prime'],
    theta=both_genre['index'],
    fill='toself',
    name='Prime',
    marker_color=PRIME_COLOR
))

fig.update_traces(fill='toself')

fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=False,
            range=[0, 0.25],
        )),
    showlegend=True
)


# fig add the title
fig.update_layout(
    title={
        'text': 'Movies genre on Netflix and Prime',
        'y': 0.9,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

# generate html file
fig.write_html('html/radar.html')
fig.show()


# %%
def plot_distribution(df: pd.DataFrame, column: str):
    """
    Plot the distribution on Netflix and Prime

    Args:
            df (pd.DataFrame): The dataframe containing the data,
            must contain the columns 'on_netflix', 'on_prime' and column
    """
    # if needed transform binary 'on_netflix' and 'on_prime' to boolean
    df['on_netflix'] = df['on_netflix'].astype(bool)
    df['on_prime'] = df['on_prime'].astype(bool)

    def _plt_hist(bins=None, is_netflix=True, log=False):
        if bins is None:
            min_value = min(df[column])
            max_value = max(df[column])
            bins = np.arange(min_value, max_value, (max_value-min_value)//20)

        plt.hist(
            df[df['on_netflix' if is_netflix else 'on_prime']][column],
            bins=bins,
            alpha=0.5,
            density=True,
            color=NETFLIX_COLOR if is_netflix else PRIME_COLOR,
            label='Netflix' if is_netflix else 'Prime',
            log=log,
        )

    if column == 'averageRating':
        _plt_hist(bins=np.arange(0, 10.1, 0.5), is_netflix=True)
        _plt_hist(bins=np.arange(0, 10.1, 0.5), is_netflix=False)

        plt.title('Movie rating distribution on Netflix and Prime')
        plt.xlabel('Average rating')

    elif column == 'runtimeMinutes':
        _plt_hist(bins=np.arange(0, 350, 10), is_netflix=True)
        _plt_hist(bins=np.arange(0, 350, 10), is_netflix=False)

        plt.title('Movie run time distribution on Netflix and Prime')
        plt.xlabel('run rime (min)')

    elif column == 'release_year':
        _plt_hist(is_netflix=True, log=True)
        _plt_hist(is_netflix=False, log=True)

        plt.title('Movie release year distribution on Netflix and Prime')
        plt.xlabel('release year')

    if column == 'revenue':
        _plt_hist(is_netflix=True, log=True)
        _plt_hist(is_netflix=False, log=True)

        plt.title('Movies revenues distribution on Netflix and Prime')
        plt.xlabel('revenue ($)')

    if column == 'numVotes':
        _plt_hist(is_netflix=True, log=True)
        _plt_hist(is_netflix=False, log=True)

        plt.title('Movies numner of votes on Netflix and Prime')
        plt.xlabel('number of votes')

    plt.axvline(df[df['on_netflix']][column].mean(),
                color=NETFLIX_COLOR, linestyle='dashed', linewidth=1)
    plt.axvline(df[df['on_netflix']][column].median(),
                color=NETFLIX_COLOR, linestyle='dotted', linewidth=1)

    plt.axvline(df[df['on_prime']][column].mean(),
                color=PRIME_COLOR, linestyle='dashed', linewidth=1)
    plt.axvline(df[df['on_prime']][column].median(),
                color=PRIME_COLOR, linestyle='dotted', linewidth=1)

    # add legend for hist and mean/median
    legend_elements = [
        plt.Line2D([0], [0], color=NETFLIX_COLOR, lw=6, label='Netflix'),
        plt.Line2D([0], [0], color=PRIME_COLOR, lw=6, label='Prime'),
        plt.Line2D([0], [0], color='k', lw=3,
                   linestyle='dashed', label='Mean'),
        plt.Line2D([0], [0], color='k', lw=3,
                   linestyle='dotted', label='Median')
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.ylabel('Density')
    plt.show()


# %%
plot_distribution(df, 'averageRating')
plot_distribution(df, 'runtimeMinutes')
plot_distribution(df, 'release_year')
plot_distribution(df, 'revenue')
plot_distribution(df, 'numVotes')


# %%
# compare p value of average rating of movies on netflix and prime
ttest_ind(netflix_movies['averageRating'], prime_movies['averageRating'],
          equal_var=False, alternative='greater')

# %% [markdown]
# > The pvalue is smaller than 1%. We can reject the null hypothesis of equal means of average ratings of movies on
# Netflix and Prime. We can even confirm that the mean of the average rating of movies on Netflix is greater than the
# mean of the average rating movies on Prime. Movies on Netfliy are generally higher rated than movies on Prime.

# %%
# only movies with runtime > 0
netflix_movies_runtime = netflix_movies[netflix_movies['runtimeMinutes'] > 0]
prime_movies_runtime = prime_movies[prime_movies['runtimeMinutes'] > 0]

ttest_ind(netflix_movies_runtime['runtimeMinutes'], prime_movies_runtime['runtimeMinutes'], equal_var=False,
          alternative='greater')

# %% [markdown]
# > The pvalue is smaller than 1%. We can reject the null hypothesis of equal means of running times of movies on
# Netflix and Prime. We can even confirm that the mean of the running times of movies on Netflix is greater than the
# mean of the running times of movies on Prime. Movies on Netfliy are generally longer than movies on Prime.

# %%
ttest_ind(netflix_movies['release_year'], prime_movies['release_year'],
          equal_var=False, alternative='greater')

# %% [markdown]
# > The pvalue is smaller than 1%. We can reject the null hypothesis of equal means of release years of movies on
# Netflix and Prime. We can even confirm that the mean of release years of movies on Netflix is greater than the mean of
# release years of movies on Prime. Thus movies are generally more recent than movies on prime.

# %%
ttest_ind(netflix_movies['revenue'],
          prime_movies['revenue'], alternative='greater')

# %% [markdown]
# > The pvalue is smaller than 1%. We can reject the null hypothesis of equal means of revenues movies on Netflix and
# Prime. We can even confirm that the mean of the revenues of movies on Netflix is greater than the mean of revenues of
# movies on Prime.

# %%
ttest_ind(netflix_movies['numVotes'],
          prime_movies['numVotes'], alternative='greater')

# %% [markdown]
# > The pvalue is smaller than 1%. We can reject the null hypothesis of equal means of number of votes for movies on
# Netflix and Prime. We can even confirm that the mean of the number of votes of movies on Netflix is greater than the
# mean of revenues of movies on Prime.

# %%
prod_countrie_netflix = netflix_movies['production_countries'].apply(lambda x: x.replace('[', '').replace(
    ']', '').replace('"', '').replace(' ', '').replace("'", '').split(',')).sum()
prod_countrie_netflix = pd.Series(
    prod_countrie_netflix).value_counts(normalize=False)
prod_countrie_prime = prime_movies['production_countries'].apply(lambda x: x.replace('[', '').replace(
    ']', '').replace('"', '').replace(' ', '').replace("'", '').split(',')).sum()
prod_countrie_prime = pd.Series(
    prod_countrie_prime).value_counts(normalize=False)

df_prod_countrie = pd.concat(
    [prod_countrie_netflix, prod_countrie_prime], axis=1)
df_prod_countrie.columns = ['Netflix', 'Prime']
df_prod_countrie = df_prod_countrie.fillna(0)

# df_prod_countrie to dataframe with columns 'production_countries' and 'Netflix' and 'Prime'
df_prod_countrie = df_prod_countrie.reset_index()
df_prod_countrie.columns = ['production_countries', 'Netflix', 'Prime']

# %%
df_prod_countrie

# %%
df_prod_countrie
# turn netflix and prime columns into one column with values 'Netflix' and 'Prime'
df_prod_countrie = df_prod_countrie.melt(id_vars=['production_countries'], value_vars=['Netflix', 'Prime'],
                                         var_name='streaming_service', value_name='size')
df_prod_countrie['size'] = df_prod_countrie['size']

# %%
# remove incorrect codes from production_countries
df_prod_countrie = df_prod_countrie[df_prod_countrie['production_countries'] != '']
df_prod_countrie = df_prod_countrie[df_prod_countrie['production_countries'] != 'SU']
df_prod_countrie = df_prod_countrie[df_prod_countrie['production_countries'] != 'YU']
df_prod_countrie = df_prod_countrie[df_prod_countrie['production_countries'] != 'XK']
df_prod_countrie = df_prod_countrie[df_prod_countrie['production_countries'] != 'XC']
df_prod_countrie = df_prod_countrie[df_prod_countrie['production_countries'] != 'AN']

# %%
# iso2 to iso3
df_prod_countrie['production_countries'] = df_prod_countrie['production_countries'].apply(
    lambda x: pycountry.countries.get(alpha_2=x).alpha_3)

# %%
fig = px.scatter_geo(df_prod_countrie, locations='production_countries', color='streaming_service',
                     hover_name='production_countries', size='size',
                     projection='natural earth')


fig.show()

fig.write_html('html/worldmap.html')

# %%
df_prod_countrie

# %%
fig = px.scatter_geo(df_prod_countrie, locations='production_countries', color='streaming_service',
                     hover_name='production_countries', size='size',
                     projection='natural earth')


fig.show()


# %%
prod_countrie_netflix = netflix_movies['production_countries'].apply(lambda x: x.replace('[', '').replace(
    ']', '').replace('"', '').replace(' ', '').replace("'", '').split(',')).sum()
prod_countrie_netflix = pd.Series(
    prod_countrie_netflix).value_counts(normalize=True)
prod_countrie_prime = prime_movies['production_countries'].apply(lambda x: x.replace(
    '[', '').replace(']', '').replace('"', '').replace(' ', '').replace("'", '').split(',')).sum()
prod_countrie_prime = pd.Series(
    prod_countrie_prime).value_counts(normalize=True)

df_prod_countrie = pd.concat(
    [prod_countrie_netflix, prod_countrie_prime], axis=1)
df_prod_countrie.columns = ['Netflix', 'Prime']
df_prod_countrie = df_prod_countrie.fillna(0)

# df_prod_countrie to dataframe with columns 'production_countries' and 'Netflix' and 'Prime'
df_prod_countrie = df_prod_countrie.reset_index()
df_prod_countrie.columns = ['production_countries', 'Netflix', 'Prime']

# # remove empty rows
# df_prod_countrie = df_prod_countrie[df_prod_countrie['production_countries'] != '']

df_prod_countrie = df_prod_countrie.sort_values(by='Prime', ascending=False)

# set a column as index
df_prod_countrie = df_prod_countrie.set_index('production_countries')
# perform bootstatrap to get confidence interval

ci_int = bootstrap((df_prod_countrie['Netflix'],), np.mean)
lower = [ci_int.confidence_interval.low]
upper = [ci_int.confidence_interval.high]
df_prod_countrie[1:30].plot(kind='bar', title='Number of movies per country in Netflix and Prime', figsize=(
    10, 5), capsize=5, color=[NETFLIX_COLOR, PRIME_COLOR])


# %%
prod_comp_netflix = netflix_movies['production_companies'].apply(lambda x: x.replace('[', '').replace(']', '').replace(
    '"', '').split(',')).sum()
prod_comp_netflix = pd.Series(prod_comp_netflix).value_counts(
    sort=True, ascending=False, normalize=True)


prod_comp_prime = prime_movies['production_companies'].apply(lambda x: x.replace('[', '').replace(']', '').replace(
    '[]', 'dk').split(',')).sum()
prod_comp_prime = pd.Series(prod_comp_prime).value_counts(
    sort=True, ascending=False, normalize=True)


# join the two dataframes to compare the production companies
df_prod_comp = pd.concat([prod_comp_netflix, prod_comp_prime], axis=1)
df_prod_comp.columns = ['Netflix', 'Prime']
df_prod_comp = df_prod_comp.fillna(0)

# get the sum of the number of movies per production company
df_prod_comp['sum'] = df_prod_comp['Netflix'] + df_prod_comp['Prime']

# sum over the column 'sum'
df_prod_comp['sum'].sum()

# sort the dataframe by the number of movies on prime
df_prod_comp = df_prod_comp.sort_values(by='sum', ascending=False)

# drop the column 'sum'
df_prod_comp = df_prod_comp.drop(columns=['sum'])

df_prod_comp[1:30].plot(kind='bar', title='Number of movies per production company in Netflix and Prime', figsize=(
    10, 5), color=[NETFLIX_COLOR, PRIME_COLOR])

# %%
# dataframe of the  with the names
df_names = pd.read_csv(
    'data/IMDb/name.basics.tsv.gz', sep='\t', compression='gzip')
writers_netflix = pd.Series(netflix_movies['writers']).value_counts(
    sort=True, ascending=False, normalize=True)
writers_prime = pd.Series(prime_movies['writers']).value_counts(
    sort=True, ascending=False, normalize=True)

df_writers = pd.concat([writers_netflix, writers_prime], axis=1)
df_writers.columns = ['Netflix', 'Prime']
df_writers = df_writers.fillna(0)

# get the sum of the number of movies per production company
df_writers['sum'] = df_writers['Netflix'] + df_writers['Prime']

# sum over the column 'sum'
df_writers['sum'].sum()

# sort the dataframe by the number of movies on prime
df_writers = df_writers.sort_values(by='sum', ascending=False)

# drop the column 'sum'
df_writers = df_writers.drop(columns=['sum'])
df_writers = df_writers.merge(
    df_names, left_index=True, right_on='nconst')

df_writers = df_writers[['primaryName', 'Netflix', 'Prime']]

df_writers[:20].set_index('primaryName').plot(
    kind='bar', title='Top 20 movie writers on Netflix and Prime', figsize=(10, 5),
    color=[NETFLIX_COLOR, PRIME_COLOR])

# set x-axis label
plt.xlabel('Writer Name')

# %%
directors_netflix = pd.Series(netflix_movies['directors']).value_counts(
    sort=True, ascending=False, normalize=True)
directors_prime = pd.Series(prime_movies['directors']).value_counts(
    sort=True, ascending=False, normalize=True)

df_directors = pd.concat([directors_netflix, directors_prime], axis=1)
df_directors.columns = ['Netflix', 'Prime']
df_directors = df_directors.fillna(0)

# get the sum of the number of movies per production company
df_directors['sum'] = df_directors['Netflix'] + df_directors['Prime']

# sum over the column 'sum'
df_directors['sum'].sum()

# sort the dataframe by the number of movies on prime
df_directors = df_directors.sort_values(by='sum', ascending=False)

# drop the column 'sum'
df_directors = df_directors.drop(columns=['sum'])
# merge df_directors with df_directors_names on the nconst and index
df_directors = df_directors.merge(
    df_names, left_index=True, right_on='nconst')
df_directors = df_directors[['primaryName', 'Netflix', 'Prime']]

# %%
# plot directors_netflix and directors_prime using plotly using the same x-axis values
_prime = df_directors[:20].Netflix
_netflix = df_directors[:20].Prime
_idx = df_directors[:20].primaryName
# plot netflix and prime on the same plot netflix using orange and blue
fig = go.Figure()
fig.add_trace(go.Bar(x=_idx, y=_prime, name='Prime', marker_color=PRIME_COLOR))
fig.add_trace(go.Bar(x=_idx, y=_netflix, name='Netflix',
              marker_color=NETFLIX_COLOR))
fig.update_layout(barmode='group')
# title of the plot
fig.update_layout(
    title_text='Top 20 movie directors on Netflix and Prime')
fig.show()
# generate html of the plot
fig.write_html('html/directors.html')

# %%
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')


# %% [markdown]
# ## NLP
# In this part we will see if we can find a different distribution of topics between movies on Netflix and Prime.
# We will find the topics using LDA.

# %% [markdown]
# ### Data Loading and Preparation

# %%
# Load data
df = prepare_df()

# %% [markdown]
# #### Tokenization

# %%
# create a new dataframe where only keep the overview, on_netflix and on_prime columns
df_overview = df[['overview', 'on_netflix', 'on_prime']].copy()
# replace the index by a range of number
df_overview.reset_index(drop=True, inplace=True)
# Tokenize the overviews
df_overview.loc[:, 'overview'] = df_overview['overview'].astype(str)
df_overview.loc[:, 'tokenized_plots'] = df_overview['overview'].apply(
    lambda movie_plot: word_tokenize(movie_plot))
df_overview.head()

# %% [markdown]
# #### Lemmatization
# we start by assocating a POS tag to each word (i.e if a word is a Noun, Verb, Adjective, etc.)

# %%
df_overview.loc[:, 'plots_with_POS_tag'] = df_overview['tokenized_plots'].apply(
    lambda tokenized_plot: pos_tag(tokenized_plot))
df_overview['plots_with_POS_tag'].head()

# %% [markdown]
# If a word has no tag we don't change it. However if there is a tag, we lemmatize the word according to its tag.

# %%
lemmatizer = WordNetLemmatizer()
# Lemmatize each word given its POS tag
df_overview.loc[:, 'lemmatized_plots'] = df_overview['plots_with_POS_tag'].apply(
    lambda tokenized_plot: [word[0] if get_wordnet_pos(word[1]) == ''
                            else lemmatizer.lemmatize(word[0], get_wordnet_pos(word[1]))
                            for word in tokenized_plot])
df_overview['lemmatized_plots'].head()

# %% [markdown]
# #### Stop words removal

# %%
# define our list of stopwords
stop_words = ['\'s']
all_stopwords = stopwords.words(
    'English') + list(string.punctuation) + stop_words

# %%
# remove the white space inside each words
df_overview.loc[:, 'plots_without_stopwords'] = df_overview['lemmatized_plots'].apply(
    lambda tokenized_plot: [word.strip() for word in tokenized_plot])
# lowercase all words in each plot
df_overview.loc[:, 'plots_without_stopwords'] = df_overview['plots_without_stopwords'].apply(
    lambda plot: [word.lower() for word in plot])
# remove stopwords from the plots
df_overview.loc[:, 'plots_without_stopwords'] = df_overview['plots_without_stopwords'].apply(
    lambda plot: [word for word in plot if word not in all_stopwords])
# remove word if contains other letter than a-z or is a single character
df_overview.loc[:, 'plots_without_stopwords'] = df_overview['plots_without_stopwords'].apply(
    lambda plot: [word for word in plot if word.isalpha() and len(word) > 1])
df_overview['plots_without_stopwords'].head()[:3]

# %% [markdown]
# #### Removing words appearing only once
# - We remove these words as they are likely to bring only noise to the model.
# - There also a lots of names which are unique, which didn't add any informations.

# %%
# compute the frequency of each word
all_words = [word for tokens in df_overview['plots_without_stopwords']
             for word in tokens]
# create a frequency distribution of each word
word_dist = nltk.FreqDist(all_words)
print(f'Number of unique words: {len(word_dist)}')
# find the words which appears only once
rare_words = [word for word, count in word_dist.items() if count == 1]
print(f'Number of words appearing once: {len(rare_words)}')
# remove words appearing only once.
df_overview.loc[:, 'plots_without_stopwords'] = df_overview['plots_without_stopwords'].apply(
    lambda plot: [word for word in plot if word not in rare_words])
df_overview['plots_without_stopwords'].head()[:3]


# %% [markdown]
# ### Latent Direchlet Allocation
# We need to create a list of tokens, i.e words that will be used inside our dictionary (depending on their frequency).
# $\\$
# We can start by creating bi-gram for some words (represent to one words by one unique composed word)
# It can be also interesting to see if creating tri-gram allows to extract more information from plots.

# %%
tokens = df_overview['plots_without_stopwords'].tolist()
bigram_model = Phrases(tokens)
tokens = list(bigram_model[tokens])
print('First movie original full overview:', df_overview['overview'].iloc[0])
print('First movie processed overview:', tokens[0])


# %% [markdown]
# We can see that our processing allows to keep only the important words, which is an essential conditions in order
# to have a performant LDA model.

# %%
def train_LDA(tokens, n_topics, n_passes, no_below, no_above, n_words):
    """ Train an LDA model given some hyperparameters
    Args:
        tokens (list): A list of lists, where each inner list is a list of tokens
        n_topics (int): The number of topics to fit the model to
        n_passes (int): The number of passes to run through the data when fitting the model
        no_below (int): The minimum number of documents a word must be in to be included in the model
        no_above (float): The maximum proportion of documents a word can be in to be included in the model
        n_words (int): The number of words to include in the list of topics.

    Returns:
        tuple: A tuple containing the trained LDA model, a list of topics, a list of topic distributions
         for each document, and the corpus used to fit the model.
    """
    dictionary, corpus = build_dictionnary_and_corpus(
        tokens, no_below=no_below, no_above=no_above)
    lda_model = create_lda_model(corpus, dictionary, n_topics)
    topics = get_topics(lda_model, n_topics, n_words)
    topic_distribution = get_topic_distribution(lda_model, corpus)
    return lda_model, topics, topic_distribution, corpus


# %% [markdown]
# #### Hyperparameters
#
# These hyperparameters were obtained by trying several combination and picking the one leading to best results.
# A Result was good if topics were differents, well described by their words and representing the different categories
# of movies in the dataframe.

# %%
no_below = 10  # minimum number of documents a word must be present in to be kept
no_above = 0.5  # maximum proportion of documents a word can be present in to be kept
n_topics = 12  # number of topics
n_passes = 120  # a good rule of thumb is n_passes = 10 * n_topics. We also seen that the LDA model converge rapidly
n_words = 30  # number of words defining each topic

# %% [markdown]
# ### Dictionnary & Corpus
# The dictionnary will be the list of unique words, and the corpus a list of movie plots bag of words.

# %% [markdown]
# #### LDA Model

# %%
lda_model, topics, topic_distribution, corpus =\
    train_LDA(tokens, n_topics, n_passes, no_below, no_above, n_words)
for i, topic in enumerate(topics):
    print('Topic {}: {}'.format(i, ' '.join(topic.split(' ')[0:12])))

# %% [markdown]
# Based the description of each topic, we could assign the following names to each one:
#
# - Topic 0: **Forces of Change**
# - Topic 1: **The Search for Purpose**
# - Topic 2: **Seeking the Truth**
# - Topic 3: **Overcoming Obstacles**
# - Topic 4: **Exploring the Unknown**
# - Topic 5: **Crime and Punishment**
# - Topic 6: **Building a Better Future**
# - Topic 7: **New Beginnings**
# - Topic 8: **Uncovering Secrets**
# - Topic 9: **Community and Connection**
# - Topic 10:**Love and Romance**
# - Topic 11:**Life's Crossroads**
#
# We see that these topics seems appropriate to descibe the movies on both platform, as they are describing variates
# categories.
#
# We print below some topic distribution for some of the overviews:

# %%
# for each movie plot, get its topic distribution (i.e the probability of each topic in descending order)
topic_distributions = get_topic_distribution(lda_model, corpus)
# plot some movies distributions
for i in [7, 11, 59]:
    print('Movie plot: {}'.format(df_overview['overview'].iloc[i]))
    print('Topic distribution for the first movie plot: {}'.format(
        topic_distributions[i][0:5]))
    print('\n')

# %% [markdown]
# ### Visualizing words of topics

# %%
# Generate a word cloud image
alice_mask = np.array(Image.open('images/wordcloud_mask/alice.png'))
love_mask = np.array(Image.open('images/wordcloud_mask/love.png'))
investigation_mask = np.array(Image.open('images/wordcloud_mask/crime.webp'))
# create subplots of 3 figures
fig, axes = plt.subplots(1, 3, figsize=(20, 10))
# generate word cloud for each topic
for i, mask in enumerate([alice_mask, love_mask, investigation_mask]):
    if i == 0:
        text = topics[1]
    elif i == 1:
        text = topics[10]
    else:
        text = topics[5]
    wc = WordCloud(background_color='white', max_words=2000,
                   mask=mask, contour_width=3, contour_color='steelblue')
    wc.generate(text)
    axes[i].imshow(wc, interpolation='bilinear')
    axes[i].axis('off')
# save the figure
fig.patch.set_facecolor('white')
plt.savefig('images/wordcloud.png', dpi=300, bbox_inches='tight')


# %% [markdown]
# #### Topics Distribution among streaming services

# %%
def get_topic_distributions_for_movies(indices, topic_distributions, n_topics):
    """
    Get the topic distributions for a list of movies.

    Args:
        indices (list): A list of movie indices.
        topic_distributions (list): A list of topic distributions for all movies.
        n_topics (int): The number of topics.

    Returns:
        list: A list of topic distributions for the specified movies.
    """
    # get the topic distribution for each movie
    movie_topic_distributions = [topic_distributions[i] for i in indices]

    # compute the total distribution for all movies
    total_distribution = [0] * n_topics
    for topic_dist in movie_topic_distributions:
        for i in range(n_topics):
            total_distribution[topic_dist[i][0]] += topic_dist[i][1]

    # divide by the total number of movies
    total_distribution = [x / len(indices) for x in total_distribution]

    return total_distribution


# %%
names = ['Forces of Change', 'The Search for Purpose', 'Seeking the Truth', 'Overcoming Obstacles',
         'Exploring the Unknown', 'Crime and Punishment', 'Building a Better Future', 'New Beginnings',
         'Uncovering Secrets', 'Community and Connection', 'Love and Romance', 'Life\'s Crossroads']

df_overview_netflix = df_overview[df_overview['on_netflix'] == 1]
df_overview_prime = df_overview[df_overview['on_prime'] == 1]


# get the topic distributions for Netflix movies
netflix_index = df_overview_netflix.index.tolist()
netflix_topics_dist = get_topic_distributions_for_movies(
    netflix_index, topic_distributions, n_topics)

# get the topic distributions for Prime movies
prime_index = df_overview_prime.index.tolist()
prime_topics_dist = get_topic_distributions_for_movies(
    prime_index, topic_distributions, n_topics)

tick_positions = np.array(range(n_topics))
bar_width = 0.4
fig = go.Figure(data=[
    go.Bar(name='Netflix', x=list(range(n_topics)), y=netflix_topics_dist, width=bar_width, marker_color=NETFLIX_COLOR,
           opacity=0.8),
    go.Bar(name='Prime', x=list(range(n_topics)), y=prime_topics_dist, width=bar_width, marker_color=PRIME_COLOR,
           opacity=0.8)
])

fig.update_layout(
    height=500,
    xaxis=dict(tickvals=tick_positions, ticktext=names, tickangle=-90),
    yaxis=dict(title='Distribution over topics'),
    title=dict(
        text='Topic distribution for movies on Netflix and Prime', x=0.45, y=0.9),
    legend_orientation='h',
    legend=dict(x=1, y=1)
)
fig.show()
# export the fig as html in html folder
fig.write_html('html/topic_distribution.html')


# %% [markdown]
# As we can see, the distribution is almost the same for each platform. This isn't so much surprising as
# we've seen before that they have roughly the same proportion of movies for most of the categories. This could also
# due to a too weak model, as trained with less data than needed.
# As the distribution are almost equals, we don't include the topic distribution during our observational analysis.
#
#
# ### NLP with sentiment analysis
#
# We can explore if movies overviews "sentiments" (positive/negative) are different between the two streaming platforms.
# Polarity is a measure of the sentiment of the overview, with values ranging from -1 (very negative) to
# 1 (very positive). Subjectivity is a measure of the objectivity of the overview, with values
# ranging from 0 (completely objective) to 1 (completely subjective).

# %%
# compute the 'sentiment' of each movie overview
netflix_blobs = [TextBlob(' '.join(plot))
                 for plot in df_overview_netflix['plots_without_stopwords']]
prime_blobs = [TextBlob(' '.join(plot))
               for plot in df_overview_prime['plots_without_stopwords']]

# Use the `sentiment` property to get the overall sentiment of each plot
netflix_sentiments = [blob.sentiment for blob in netflix_blobs]
prime_sentiments = [blob.sentiment for blob in prime_blobs]

prime_sentiments_polarity = [
    sentiment.polarity for sentiment in prime_sentiments]
prime_sentiments_subjectivity = [
    sentiment.subjectivity for sentiment in prime_sentiments]
netflix_sentiments_polarity = [
    sentiment.polarity for sentiment in netflix_sentiments]
netflix_sentiments_subjectivity = [
    sentiment.subjectivity for sentiment in netflix_sentiments]


# we plot the distribution of the polarity and subjectivity of the sentiment, for each platform
fig = make_subplots(rows=1, cols=2,
                    subplot_titles=('Sentiment polarity distribution', 'Sentiment subjectivity distribution'))
data = [
    go.Histogram(name='Netflix', x=netflix_sentiments_polarity, marker_color=NETFLIX_COLOR, opacity=0.5,
                 histnorm='probability', xbins=dict(start=-1, end=1, size=0.1), legendgroup='Netflix'),
    go.Histogram(name='Prime', x=prime_sentiments_polarity, marker_color=PRIME_COLOR, opacity=0.5,
                 histnorm='probability', xbins=dict(start=-1, end=1, size=0.1)),
    go.Histogram(name='Netflix', x=netflix_sentiments_subjectivity, marker_color=NETFLIX_COLOR, opacity=0.5,
                 histnorm='probability', xbins=dict(start=0, end=1, size=0.05), showlegend=False,
                 legendgroup='Netflix'),
    go.Histogram(name='Prime', x=prime_sentiments_subjectivity, marker_color=PRIME_COLOR, opacity=0.5,
                 histnorm='probability', xbins=dict(start=0, end=1, size=0.05), showlegend=False),
]
for i, elem in enumerate(data):
    fig.add_trace(elem, row=1, col=int(1 + i/2))

# compute mean and median for each distribution for each service
netflix_polarity_mean = np.mean(netflix_sentiments_polarity)
netflix_polarity_median = np.median(netflix_sentiments_polarity)
prime_polarity_mean = np.mean(prime_sentiments_polarity)
prime_polarity_median = np.median(prime_sentiments_polarity)
netflix_subjectivity_mean = np.mean(netflix_sentiments_subjectivity)
netflix_subjectivity_median = np.median(netflix_sentiments_subjectivity)
prime_subjectivity_mean = np.mean(prime_sentiments_subjectivity)
prime_subjectivity_median = np.median(prime_sentiments_subjectivity)
# polarity mean and median
polarity_y_range = [0, 0.31]
subjectivity_y_range = [0, 0.12]
width = 1
mean_median_lines = [
    go.Scatter(x=[netflix_polarity_mean]*2, y=polarity_y_range, name='Netflix Mean', legendgroup='Mean',
               mode='lines', line=dict(color=NETFLIX_COLOR, width=width, dash='dash'), showlegend=False),
    go.Scatter(x=[netflix_polarity_median]*2, y=polarity_y_range, name='Netflix Median', legendgroup='Median',
               mode='lines', line=dict(color=NETFLIX_COLOR, width=width, dash='dot'), showlegend=False),
    go.Scatter(x=[prime_polarity_mean]*2, y=polarity_y_range, mode='lines', name='Prime Mean',
               line=dict(color=PRIME_COLOR, width=width, dash='dash'), legendgroup='Mean', showlegend=False),
    go.Scatter(x=[prime_polarity_median]*2, y=polarity_y_range, legendgroup='Median', showlegend=False,
               name='Prime Median', mode='lines', line=dict(color=PRIME_COLOR, width=width, dash='dot')),
    go.Scatter(x=[netflix_subjectivity_mean]*2, y=subjectivity_y_range, legendgroup='Mean', name='Netflix Mean',
               showlegend=False, mode='lines', line=dict(color=NETFLIX_COLOR, width=width, dash='dash')),
    go.Scatter(x=[netflix_subjectivity_median]*2, y=subjectivity_y_range, legendgroup='Median', name='Netflix Median',
               showlegend=False, mode='lines', line=dict(color=NETFLIX_COLOR, width=width, dash='dot')),
    go.Scatter(x=[prime_subjectivity_mean]*2, y=subjectivity_y_range, legendgroup='Mean', name='Prime Mean',
               showlegend=False, mode='lines', line=dict(color=PRIME_COLOR, width=width, dash='dash')),
    go.Scatter(x=[prime_subjectivity_median]*2, y=subjectivity_y_range, legendgroup='Median', name='Prime Median',
               showlegend=False, mode='lines', line=dict(color=PRIME_COLOR, width=width, dash='dot')),
]
for i, elem in enumerate(mean_median_lines):
    fig.add_trace(elem, row=1, col=int(1 + i/4))

# legends purpose only
line_legend = [
    go.Scatter(
        yaxis='y2',
        x=[0],
        y=[0],
        name='Median',
        mode='lines',
        line=dict(color='black', width=1, dash='dot'),
        legendgroup='Median',
    ),
    go.Scatter(
        yaxis='y2',
        x=[0],
        y=[0],
        name='Mean',
        mode='lines',
        line=dict(color='black', width=1, dash='dash'),
        legendgroup='Mean',
    )
]
for elem in line_legend:
    fig.add_trace(elem, row=1, col=1)

fig.update_layout(barmode='overlay')
for i, name in enumerate(['Polarity', 'Subjectivity'], start=1):
    fig.update_xaxes(title_text='{} value'.format(name), row=1, col=i)
    fig.update_yaxes(title_text='probability', row=1, col=i)
# export the fig as html in html folder
fig.write_html('html/sentiments.html')
fig.show()

# %% [markdown]
# From these graphs, we see that the polarity is a little bit higher than 0 for Netflix. It
# also seems that Prime have movies with a lower polarity than Netflix, i.e. Prime movies overviews would have a
# more "negative" sentiment. For subjectivity, we see both distributions seems roughly equal. We also see
# that Prime seems to have more "objective" movies overview than Neflix. Netflix on its side seems to have a little bit
# more "subjective" overviews.

# %%
# Perform a t-test to compare the means of the polarity distributions
t_statistic_polarity, p_value_polarity = ttest_ind(
    prime_sentiments_polarity, netflix_sentiments_polarity)
t_statistic_subjectivity, p_value_subjectivity = ttest_ind(
    prime_sentiments_subjectivity, netflix_sentiments_subjectivity)

print('Polarity p-value ', round(p_value_polarity, 5))
print('Subjectivity p-value', round(p_value_subjectivity, 4))


# %% [markdown]
# As the polarity p-value is less than 0.05, it suggests that the difference between the means of the two distributions
# is statistically significant, and we can reject the null hypothesis that the two distributions are similar.
# We see that this analysis is true for both polarity and subjectivity.
#
# We will then see in the following of this project, if polarity of overviews can help to determine
# which streaming service is the best.
# We didn't keep the subjectivity for the following reason:
# One idea would have been to select only overviews with low subjectivity score (i.e. objective reviews). This would
# have prevent rating of the movies be influenced by a biased overview. However, by looking at the subjectivity
# distribution, this would have leave us with less than half of the movies which in term of robustness
# wasn't the best choice as we already conduct our study on roughly ten thousands movies.
#
# #### Prepare Dataframe for Observational Study

# %%
# put into a new dataframa a copy of df_overview_prime
df_overview_prime_save = df_overview_prime.copy()

# %%
df_overview_prime.loc[:, 'sentiments_polarity'] = prime_sentiments_polarity

# %%
# add the prime_sentiments_polarity and netflix_sentiments_polarity to the dataframe
df_overview_prime.loc[:, 'sentiments_polarity'] = prime_sentiments_polarity
df_overview_netflix.loc[:, 'sentiments_polarity'] = netflix_sentiments_polarity
# add the prime_sentiments_subjectivity and netflix_sentiments_subjectivity to the dataframe
df_overview_prime.loc[:,
                      'sentiments_subjectivity'] = prime_sentiments_subjectivity
df_overview_netflix.loc[:,
                        'sentiments_subjectivity'] = netflix_sentiments_subjectivity
# remove the tokenized plots, plots_with_POS_tag, lemmatized_plots, plots_without_stopwords columns
# as we don't need them for observational study
df_overview_prime.drop(
    ['tokenized_plots', 'plots_with_POS_tag', 'lemmatized_plots', 'plots_without_stopwords'], axis=1, inplace=True)
df_overview_netflix.drop(
    ['tokenized_plots', 'plots_with_POS_tag', 'lemmatized_plots', 'plots_without_stopwords'], axis=1, inplace=True)
# concatenate the two dataframes
df_overview = pd.concat([df_overview_prime, df_overview_netflix])
df_overview.head()

# %% [markdown]
# # Observational Study
#
# As we have seen in our exploration part, some features aren't equally distributed between Netflix and Prime. The
# basic rating comparison obtained in exploration part between Prime and Netflix may in consequence be biased,
# as some features of a movie other
# than appartenance to Netflix or Prime may have an influence on the movie rating. In order to unbiased our analysis,
# we will reduce as much as possible the difference of features between movies on Netflix and Prime. We will achieve
# this by doing a matching.
#
# We start by adding our result from NLP (i.e. movie sentiment polarity) in our dataframe

# %%

df = helper.prepare_df()
df['genres'] = df['genres'].apply(lambda x: x.split(','))
df.reset_index(drop=True, inplace=True)
# merge the original dataframe and the one containing the sentiment analysis
merged_df = df.merge(df_overview, left_index=True,
                     right_index=True, suffixes=('', '_df2'))
# Get the list of common columns between the two dataframes
common_columns = list(set(df.columns) & set(df_overview.columns))

# Keep only the column from the first dataframe for each common column
for col in common_columns:
    merged_df[col] = merged_df[col]
    merged_df.drop(columns=[col + '_df2'], inplace=True)

merged_df = merged_df[~merged_df.index.duplicated(keep='first')]
df = merged_df.copy()
print(len(df))

# %% [markdown]
# ### Data Cleaning
#
# We check if we have some undefined values in the dataframe.

# %%
# for each column, if there are missing values, print the number of missing values and remove the line.
for col in df.columns:
    if (df[col].isna().sum()) > 0:
        print(f'{col} has {df[col].isna().sum()} missing values')
        # remove the line with missing values
        df.dropna(subset=[col], inplace=True)

# %% [markdown]
# We decided to drop the line as replacing them with mean of runtime could biaise our data, and the number
# of missing value was around 2% of the total data.

# %% [markdown]
# # Naive analysis
# In this first part, we will try to understand the data by looking at the distribution of the different variables
# without doing some matching or any special preprocessing.
#

# %%
df['streaming_service'] = ['both' if netflix and prime else 'netflix' if netflix
                           else 'prime' for netflix, prime in zip(df['on_netflix'], df['on_prime'])]


# %%
def plot_rating_distribution(df: pd.DataFrame, n: int):
    """
    Plot the rating distribution on Netflix and Prime using Plotly.

    Args:
        df (pd.DataFrame): The dataframe containing the data,
            must contain the columns 'on_netflix', 'on_prime' and 'averageRating'
        n (int): indicator of html version. 1 means before matching, 2 means after matching,
         3 means after country of prod. matching
    """
    # if needed transform binary 'on_netflix' and 'on_prime' to boolean
    df['on_netflix'] = df['on_netflix'].astype(bool)
    df['on_prime'] = df['on_prime'].astype(bool)

    traces = [
        go.Histogram(
            x=df[df['on_netflix']]['averageRating'],
            histnorm='probability',
            xbins=dict(start=0, end=10, size=0.1),
            opacity=0.5,
            name='Netflix',
            marker_color=NETFLIX_COLOR,
        ),
        go.Histogram(
            x=df[df['on_prime']]['averageRating'],
            histnorm='probability',
            xbins=dict(start=0, end=10, size=0.1),
            opacity=0.5,
            name='Prime',
            marker_color=PRIME_COLOR
        ),
        go.Scatter(
            x=[df[df['on_netflix']]['averageRating'].mean()] * 2,
            y=[0, 1],
            yaxis='y2',
            name='Netflix mean',
            mode='lines',
            line=dict(color=NETFLIX_COLOR, width=1, dash='dash'),
            legendgroup='Mean',
            showlegend=False,
        ),
        go.Scatter(
            x=[df[df['on_netflix']]['averageRating'].median()] * 2,
            y=[0, 1],
            yaxis='y2',
            name='Netflix median',
            mode='lines',
            line=dict(color=NETFLIX_COLOR, width=1, dash='dot'),
            legendgroup='Median',
            showlegend=False,
        ),
        go.Scatter(
            x=[df[df['on_prime']]['averageRating'].mean()] * 2,
            y=[0, 1],
            yaxis='y2',
            name='Prime mean',
            mode='lines',
            line=dict(color=PRIME_COLOR, width=1, dash='dash'),
            legendgroup='Mean',
            showlegend=False,
        ),
        go.Scatter(
            x=[df[df['on_prime']]['averageRating'].median()] * 2,
            y=[0, 1],
            yaxis='y2',
            name='Prime median',
            mode='lines',
            line=dict(color=PRIME_COLOR, width=1, dash='dot'),
            legendgroup='Median',
            showlegend=False,
        ),
        # legends purpose only
        go.Scatter(
            yaxis='y2',
            x=[0],
            y=[0],
            name='Median',
            mode='lines',
            line=dict(color='black', width=1, dash='dot'),
            legendgroup='Median',
        ),
        go.Scatter(
            yaxis='y2',
            x=[0],
            y=[0],
            name='Mean',
            mode='lines',
            line=dict(color='black', width=1, dash='dash'),
            legendgroup='Mean',
        ),
    ]

    fig = go.Figure(data=traces)

    fig.update_layout(barmode='overlay')

    fig.update_layout(
        title='Movie rating distribution on Netflix and Prime',
        title_x=0.5,
        xaxis_title='Average rating',
        yaxis_title='Probability',
        yaxis2=dict(
            overlaying='y',
            showgrid=False,
            showline=False,
            showticklabels=False,
            range=[0, 1],
        ),

        legend=dict(
            yanchor='top',
            y=0.99,
            xanchor='left',
            x=0.01
        )
    )

    # export the fig as html in html folder
    if n == 1:
        fig.write_html('html/ratings_before_matching.html')
    elif n == 2:
        fig.write_html('html/ratings_after_matching.html')
    elif n == 3:
        fig.write_html('html/ratings_after_matching_countries.html')
    else:
        raise ValueError('n should be 1, 2 or 3')
    # show the plots
    fig.show()


# %%
plot_rating_distribution(df, 1)


# %%
def ttest_ratings(df: pd.DataFrame) -> None:
    """Perform a ttest to compare the average rating of movies on Netflix and Prime.
    Use the greater alternative hypothesis.

    Args:
        df (pd.DataFrame): dataframe containing the data,
        must contain the columns 'on_netflix', 'on_prime' and 'averageRating'
    """
    t_statistic_rating, p_value_rating = ttest_ind(
        df[df['on_netflix']]['averageRating'], df[df['on_prime']]['averageRating'],
        equal_var=False, alternative='greater')
    print(
        f'averageRating: t-statistic: {round(t_statistic_rating, 2)}, p-value: {round(p_value_rating,2)}')


# %%
ttest_ratings(df)

# %% [markdown]
# As we can see from the test, the p-value is lower than our 0.05 treshold. As we are testing the 'greater' alternative,
# this means that Netflix has better ratings than Prime.
# From the graph we could also infer that you are more likely able to find an high rated movie on Netlfix
# than on Amazon Prime.
#
# Now we will experience a more robust comparison.

# %% [markdown]
# # Matching based on propensity score
# First we'll create an adapted dataframe with only the variables we want to use for the matching.
#
# From exploration part, we decided to keep the following movie features:
# - averageRating
# - numVotes
# - release_year
# - runtimeMinutes
# - sentiments_polarity
# - genres
# - production_countries
#

# %%
df.columns

# %%
matching_df = df[['averageRating', 'numVotes', 'release_year',
                  'runtimeMinutes', 'genres', 'on_netflix', 'on_prime',
                  'sentiments_polarity', 'production_countries']].copy()
# we only keep numbers or values that we'll be able to transform to binary

matching_df['on_netflix'] = matching_df['on_netflix'].apply(
    lambda x: 1 if x else 0)
matching_df['on_prime'] = matching_df['on_prime'].apply(
    lambda x: 1 if x else 0)

# %%
# create a list of genres with their occurences
genres = dict()
for genre in matching_df['genres']:
    for g in genre:
        if g in genres:
            genres[g] += 1
        else:
            genres[g] = 1

# remove \N
genres.pop('\\N')

# create a list sorted by occurrences, keeping only the genres
genres = [genre for genre, _ in sorted(
    genres.items(), key=lambda item: -item[1])]

genres


# %%
mlb = MultiLabelBinarizer(classes=genres)

genres_df = pd.DataFrame(mlb.fit_transform(
    matching_df['genres']), columns=mlb.classes_, index=matching_df.index)
matching_df = pd.concat([matching_df, genres_df], axis=1)
matching_df.drop(columns=['genres'], inplace=True)
matching_df.head()

# %%
# pairplot but only with averageRating
sns.pairplot(df[['averageRating', 'numVotes',
                 'release_year', 'runtimeMinutes', 'streaming_service', 'sentiments_polarity']],
             hue='streaming_service', palette=[NETFLIX_COLOR, PRIME_COLOR, 'green'])
plt.show()

# %%
df_netflix = matching_df[matching_df['on_netflix'] == 1].copy()
df_netflix.drop(columns=['on_netflix', 'on_prime'], inplace=True)

df_prime = matching_df[matching_df['on_prime'] == 1].copy()
df_prime.drop(columns=['on_netflix', 'on_prime'], inplace=True)


# %%
def compute_log_bins(df, num_bins, min_val, max_val):
    """Compute the bins for a logarithmic histogram.

    Args:
        df (pd.DataFrame): The dataframe containing the data.
        num_bins (int): The number of bins.
        min_val (float): The minimum value of the data.
        max_val (float): The maximum value of the data.

    Returns:
        position_array: The position of the bars.
        height_array: The height of the bars.
        width_array: The width of the bars.
    """

    # Compute the logarithmically spaced bins
    log_bins = np.logspace(np.log10(min_val), np.log10(max_val), num_bins+1)

    # Compute the position, height, and width arrays for the bars
    position_array = (log_bins[:-1] + log_bins[1:]) / 2
    # we used np.digitize to split our data into bins
    height_array = df.groupby(np.digitize(df, log_bins)).count().values
    width_array = log_bins[1:] - log_bins[:-1]

    return position_array, height_array, width_array


# %%
def plot_hist_matching(df_netflix: pd.DataFrame, df_prime: pd.DataFrame, n: int):
    """
    Plot the histogram of the matching features on Netflix and Prime

    Args:
        df_netflix (pd.DataFrame): The dataframe containing the data for Netflix
        df_prime (pd.DataFrame): The dataframe containing the data for Prime
        n (int): indicator of html version. 1 means before matching, 2 means after matching,
         3 means after country of prod. matching
    """

    columns_to_plot = ['numVotes', 'release_year',
                       'runtimeMinutes', 'sentiments_polarity']
    columns_names = ['Number of votes', 'Release year',
                     'Runtime (minutes)', 'Sentiments polarity']

    fig = make_subplots(rows=2, cols=2, subplot_titles=columns_names)

    for i, col in enumerate(columns_to_plot):
        if col in ['release_year', 'runtimeMinutes', 'sentiments_polarity']:
            histograms = [
                go.Histogram(
                    x=df_netflix[col],
                    name='Netflix',
                    histnorm='probability',
                    legendgroup='Netflix',
                    marker_color=NETFLIX_COLOR,
                    bingroup=col,
                    showlegend=False,
                ),
                go.Histogram(
                    x=df_prime[col],
                    name='Prime',
                    histnorm='probability',
                    legendgroup='Prime',
                    marker_color=PRIME_COLOR,
                    bingroup=col,
                    showlegend=False,
                ),
            ]

            for histogram in histograms:
                fig.add_trace(histogram, row=i//2+1, col=i % 2+1)

        elif col in ['numVotes']:
            min_val = min(df_netflix[col].min(), df_prime[col].min())
            max_val = max(df_netflix[col].max(), df_prime[col].max())

            bars_positions_netflix, bins_height_netflix, bins_width_netflix = compute_log_bins(
                df_netflix[col], 40, min_val, max_val)
            bars_positions_prime, bins_height_prime, bins_width_prime = compute_log_bins(
                df_prime[col], 40, min_val, max_val)

            bars = [
                go.Bar(
                    x=bars_positions_netflix,
                    y=bins_height_netflix,
                    name='Netflix',
                    legendgroup='Netflix',
                    marker_color=NETFLIX_COLOR,
                    width=bins_width_netflix,
                ),
                go.Bar(
                    x=bars_positions_prime,
                    y=bins_height_prime,
                    name='Prime',
                    legendgroup='Prime',
                    marker_color=PRIME_COLOR,
                    width=bins_width_prime,
                ),
            ]

            for bar in bars:
                fig.add_trace(bar, row=i//2+1, col=i % 2+1)

            fig.update_xaxes(type='log', row=i//2+1, col=i % 2+1)

        else:
            # should never happen
            raise ValueError('column not found')

    fig.update_layout(barmode='overlay')
    fig.update_traces(opacity=0.5)
    if n == 1:
        fig.write_html('html/features_before_matching.html')
    elif n == 2:
        fig.write_html('html/features_after_matching.html')
    elif n == 3:
        fig.write_html('html/features_after_matching_countries.html')
    else:
        raise ValueError('n should be 1, 2 or 3')
    fig.show()


plot_hist_matching(df_netflix, df_prime, 1)
# compute some statistics to see if the distributions are different
for col in ['numVotes', 'runtimeMinutes', 'sentiments_polarity', 'release_year']:
    t_statistic_polarity, p_value_polarity = ttest_ind(
        df_prime[col], df_netflix[col], equal_var=False)
    print(f'{col}: t-statistic: {round(t_statistic_polarity, 2)}, p-value: {round(p_value_polarity,2)}')


# %% [markdown]
# We see that all p-values are lower than the treshold of 0.05. We can reject our null hypothesis that the distibutions
# of netflix and prime are equal for each feature. We will see if we can matched the film between prime and netflix
# to obtain the same distribution for each feature.

# %%
def plot_genre_distribution(df_netflix: pd.DataFrame, df_prime: pd.DataFrame, n: int):
    """Plot the genre distribution on Netflix and Prime

    Args:
        df_netflix (pd.DataFrame): The dataframe containing the data for Netflix
        df_prime (pd.DataFrame): The dataframe containing the data for Prime
        n (int): indicator of html version. 1 means before matching, 2 means after matching,
         3 means after country of prod. matching
    """
    # for each genre, compute the number of movies on Netflix and Prime
    netflix_categories_dist = df_netflix[genres].sum() / len(df_netflix)
    prime_categories_dist = df_prime[genres].sum() / len(df_prime)
    netflix_categories_dist = netflix_categories_dist.sort_values(
        ascending=False)
    prime_categories_dist = prime_categories_dist.sort_values(ascending=False)
    x_positions = np.arange(len(genres))

    bar_width = 0.35
    fig = go.Figure(data=[
        go.Bar(name='Netflix', x=x_positions, y=netflix_categories_dist*100, width=bar_width,
               marker_color=NETFLIX_COLOR, opacity=0.8),
        go.Bar(name='Prime', x=x_positions, y=prime_categories_dist*100, width=bar_width, marker_color=PRIME_COLOR,
               opacity=0.8)
    ])

    fig.update_layout(
        height=500,
        xaxis=dict(tickvals=x_positions, ticktext=genres, tickangle=-90),
        yaxis=dict(title='% of Movies'),
        title=dict(
            text='% of Movies per Genre for each Streaming Service', x=0.45, y=0.9),
        legend_orientation='h',
        legend=dict(x=1, y=1)
    )

    # export to html
    if n == 1:
        fig.write_html('html/genres_before_matching.html')
    elif n == 2:
        fig.write_html('html/genres_after_matching.html')
    elif n == 3:
        fig.write_html('html/genres_after_matching_countries.html')
    else:
        raise ValueError('n should be 1, 2 or 3')
    fig.show()


plot_genre_distribution(df_netflix, df_prime, 1)

# %% [markdown]
# We'll now compute a propensity score for each observation using a logistic regression.

# %%
# we don't normalize the sentiments_polarity as it is already between -1 and 1
features_to_normalize = ['numVotes', 'release_year', 'runtimeMinutes']
for feature in features_to_normalize:
    matching_df['normalized_' + feature] = (
        matching_df[feature] - matching_df[feature].mean()) / matching_df[feature].std()

# %%
matching_df.columns

# %%
# we train a random forest classifier in order to predict if a movie is on Netflix or not
X = matching_df[['normalized_numVotes', 'normalized_release_year',
                 'normalized_runtimeMinutes', 'sentiments_polarity']]
y = matching_df['on_netflix']
# we use a large test data set to have a good estimate of the performance and try to avoid overfitting
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42)
# we use a low max_depth to avoid overfitting
rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
rf.fit(X_train, y_train)
matching_df['predicted_netflix'] = rf.predict_proba(X)[:, 1]
print(classification_report(y_test, rf.predict(X_test)))
# plot the ROC curve
fpr, tpr, thresholds = roc_curve(y, matching_df['predicted_netflix'])
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc='lower right')
plt.show()

# %% [markdown]
# As we can see our model achieve decent performances, as we have a ROC curve area of 0.93.

# %%
# update df with the predicted probability of being on netflix
# for this part we keep only movies that are either only on netflix or only on prime
mask_netflix_only = (matching_df['on_netflix'] == 1) & (
    matching_df['on_prime'] == 0)
mask_prime_only = (matching_df['on_netflix'] == 0) & (
    matching_df['on_prime'] == 1)

df_netflix = matching_df[mask_netflix_only].copy()
df_netflix.drop(columns=['on_netflix', 'on_prime'], inplace=True)

df_prime = matching_df[mask_prime_only].copy()
df_prime.drop(columns=['on_netflix', 'on_prime'], inplace=True)


# %%
def get_similarity(propensity_score1, propensity_score2):
    """Calculate similarity with given propensity scores

    Args:
        propensity_score1 (float): propensity score of first movie
        propensity_score2 (float): propensity score of second movie

    Returns:
        float: similarity between 0 and 1
    """
    return 1-np.abs(propensity_score1-propensity_score2)

# %%
# TODO remove for submission
# only keep 20%
# df_netflix = df_netflix.sample(frac=0.2)
# df_prime = df_prime.sample(frac=0.2)


# %%
# create a set of genres for each movie, to make comparisons faster (>50x faster)
df_netflix['genres'] = df_netflix[genres].apply(
    lambda x: set(x[x == 1].index), axis=1)
df_prime['genres'] = df_prime[genres].apply(
    lambda x: set(x[x == 1].index), axis=1)

G = nx.Graph()

for netflix_id, netflix_row in tqdm(df_netflix.iterrows(), total=df_netflix.shape[0]):
    for prime_id, prime_row in df_prime.iterrows():

        # (less edges in the graph, faster computation)
        # here we put some conditions to avoid adding edges between instances that are too different

        # if genres are different, skip
        if netflix_row['genres'] != prime_row['genres']:
            continue

        # Calculate the similarity
        similarity = get_similarity(netflix_row['predicted_netflix'],
                                    prime_row['predicted_netflix'])

        # we want a similarity of at least 0.5
        if similarity < 0.5:
            continue

        # Add an edge between the two instances weighted by the similarity between them
        G.add_weighted_edges_from([(netflix_id, prime_id, similarity)])


# %%
print(f'nb of nodes: {G.number_of_nodes()}')
print(f'nb of edges: {G.number_of_edges()}')

# %%
# can be really long
matching = nx.max_weight_matching(G)

# %%
matched = [elem for tuple_elem in list(matching) for elem in tuple_elem]

# %%
balanced_df = matching_df.loc[matched]

df_netflix = balanced_df[balanced_df['on_netflix'] == 1].copy()
df_netflix.drop(columns=['on_netflix', 'on_prime'], inplace=True)

df_prime = balanced_df[balanced_df['on_prime'] == 1].copy()
df_prime.drop(columns=['on_netflix', 'on_prime'], inplace=True)

# %%
len(balanced_df)

# %%
df_netflix.columns

# %%
plot_hist_matching(df_netflix, df_prime, 2)
for col in ['numVotes', 'runtimeMinutes', 'sentiments_polarity', 'release_year']:
    t_statistic_polarity, p_value_polarity = ttest_ind(
        df_prime[col], df_netflix[col], equal_var=False)
    print(f'{col}: t-statistic: {round(t_statistic_polarity, 2)}, p-value: {round(p_value_polarity,2)}')

# %% [markdown]
# We see that for runtime,sentiments polarity and numVotes, the p-value is greater than some threshold of 0.05. Thus we
# cannot reject the null hypothesis that the two distributions are equals. For release_year the p-value
# is still lower than the threshold. We can reject the null hypothesis, so the two distributions are still unequal.
# This can be expected as we try to match on a lots of parameters, and we don't make a perfect matching, so it's still
# normal to observe some drift between the netflix and prime distributions. We can however see that even if
# distributions are not equals for realease_year, they seems more similar than prior matching. We have seen that the
# matching actually allows to have similar distribution of features over prime and netflix. This allows a more rigourous
# comparison between the rating of movies on both streaming services.

# %%
plot_genre_distribution(df_netflix, df_prime, 2)

# %%
plot_rating_distribution(balanced_df, 2)

# %%
ttest_ratings(balanced_df)

# %% [markdown]
# # TODO analysis of results

# %% [markdown]
# We now redo the same analysis as before, but this time we'll do a perfect matching also on the production country.

# %%
mask_netflix_only = (matching_df['on_netflix'] == 1) & (
    matching_df['on_prime'] == 0)
mask_prime_only = (matching_df['on_netflix'] == 0) & (
    matching_df['on_prime'] == 1)

df_netflix = matching_df[mask_netflix_only].copy()
df_netflix.drop(columns=['on_netflix', 'on_prime'], inplace=True)

df_prime = matching_df[mask_prime_only].copy()
df_prime.drop(columns=['on_netflix', 'on_prime'], inplace=True)

# TODO remove for final submission
# only keep 20%
# df_netflix = df_netflix.sample(frac=0.2)
# df_prime = df_prime.sample(frac=0.2)

# create a set of genres for each movie, to make comparisons faster (>50x faster)
df_netflix['genres'] = df_netflix[genres].apply(
    lambda x: set(x[x == 1].index), axis=1)
df_prime['genres'] = df_prime[genres].apply(
    lambda x: set(x[x == 1].index), axis=1)

# add production country as requirement for the matching
G = nx.Graph()

for netflix_id, netflix_row in tqdm(df_netflix.iterrows(), total=df_netflix.shape[0]):
    for prime_id, prime_row in df_prime.iterrows():

        # (less edges in the graph, faster computation)
        # here we put some conditions to avoid adding edges between instances that are too different

        # if production country is different, skip
        if netflix_row['production_countries'] != prime_row['production_countries']:
            continue

        # if genres are different, skip
        if netflix_row['genres'] != prime_row['genres']:
            continue

        # Calculate the similarity
        similarity = get_similarity(netflix_row['predicted_netflix'],
                                    prime_row['predicted_netflix'])

        # we want a similarity of at least 0.5
        if similarity < 0.5:
            continue

        # Add an edge between the two instances weighted by the similarity between them
        G.add_weighted_edges_from([(netflix_id, prime_id, similarity)])

# %%
print(f'nb of nodes: {G.number_of_nodes()}')
print(f'nb of edges: {G.number_of_edges()}')

# %%
# can be really long
matching = nx.max_weight_matching(G)

# %%
matched = [elem for tuple_elem in list(matching) for elem in tuple_elem]

balanced_df = matching_df.loc[matched]

df_netflix = balanced_df[balanced_df['on_netflix'] == 1].copy()
df_netflix.drop(columns=['on_netflix', 'on_prime'], inplace=True)

df_prime = balanced_df[balanced_df['on_prime'] == 1].copy()
df_prime.drop(columns=['on_netflix', 'on_prime'], inplace=True)

len(balanced_df)

# %%
plot_hist_matching(df_netflix, df_prime, 3)
for col in ['numVotes', 'runtimeMinutes', 'sentiments_polarity', 'release_year']:
    t_statistic_polarity, p_value_polarity = ttest_ind(
        df_prime[col], df_netflix[col], equal_var=False)
    print(f'{col}: t-statistic: {round(t_statistic_polarity, 2)}, p-value: {round(p_value_polarity,2)}')

# %%
plot_rating_distribution(balanced_df, 3)

# %%
ttest_ratings(balanced_df)

# %% [markdown]
# # TODO
# analyze result and conclude

# %% [markdown]
#
