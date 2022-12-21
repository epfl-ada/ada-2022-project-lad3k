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
#     display_name: ada_project
#     language: python
#     name: python3
# ---
# %%
import math
import string

import matplotlib.pyplot as plt
import networkx as nx
import nltk
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import seaborn as sns
import statsmodels.formula.api as smf
from gensim.models import Phrases
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from plotly.subplots import make_subplots
from scipy.stats import ttest_ind
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from textblob import TextBlob
from tqdm import tqdm

from src import helper
from src.helper import prepare_df
from src.nlp_helper import build_dictionnary_and_corpus
from src.nlp_helper import create_lda_model
from src.nlp_helper import get_topic_distribution
from src.nlp_helper import get_topics
from src.nlp_helper import get_wordnet_pos


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
df = prepare_df()

# %%
# create a new dataframe where only keep the overview column on_netflix and on_prime
df_overview_complete = df[['overview', 'on_netflix', 'on_prime']]
df_overview_complete.head()
# replace the index by a range of number
df_overview_complete = df_overview_complete.reset_index(drop=True)
df_overview_complete.head()

# %%
df_overview = df_overview_complete.sample(frac=1.0, random_state=42)
print(f'Number of movies in the sample: {len(df_overview)}')
# convert the overview column to string
df_overview['overview'] = df_overview['overview'].astype(str)
# Tokenize the overviews
df_overview['tokenized_plots'] = df_overview['overview'].apply(
    lambda movie_plot: word_tokenize(movie_plot))
df_overview.head()['tokenized_plots']

# %% [markdown]
# #### Lemmatization
# we start by assocating a POS tag to each word (i.e if a word is a Noun, Verb, Adjective, etc.)

# %%
df_overview['plots_with_POS_tag'] = df_overview['tokenized_plots'].apply(
    lambda tokenized_plot: pos_tag(tokenized_plot))
df_overview['plots_with_POS_tag'].head()

# %% [markdown]
# If a word has no tag we don't change it. However if there is a tag, we lemmatize the word according to its tag.

# %%
lemmatizer = WordNetLemmatizer()
# Lemmatize each word given its POS tag
df_overview['lemmatized_plots'] = df_overview['plots_with_POS_tag'].apply(
    lambda tokenized_plot: [word[0] if get_wordnet_pos(word[1]) == ''
                            else lemmatizer.lemmatize(word[0], get_wordnet_pos(word[1]))
                            for word in tokenized_plot])
df_overview['lemmatized_plots'].head()

# %% [markdown]
# #### Stop words removal

# %%
# list of stop words could be improved
stop_words = ['\'s']
all_stopwords = stopwords.words(
    'English') + list(string.punctuation) + stop_words

# %%
# remove the white space inside each words
df_overview['plots_without_stopwords'] = df_overview['lemmatized_plots'].apply(
    lambda tokenized_plot: [word.strip() for word in tokenized_plot])
# lowercase all words in each plot
df_overview['plots_without_stopwords'] = df_overview['plots_without_stopwords'].apply(
    lambda plot: [word.lower() for word in plot])
# remove stopwords from the plots
df_overview['plots_without_stopwords'] = df_overview['plots_without_stopwords'].apply(
    lambda plot: [word for word in plot if word not in all_stopwords])
# remove word if contains other letter than a-z or is a single character
df_overview['plots_without_stopwords'] = df_overview['plots_without_stopwords'].apply(
    lambda plot: [word for word in plot if word.isalpha() and len(word) > 1])
df_overview['plots_without_stopwords'].head()[0:3]

# %%
# compute the frequency of each word
all_words = [word for tokens in df_overview['plots_without_stopwords']
             for word in tokens]
# create a frequency distribution of each word
word_dist = nltk.FreqDist(all_words)
print(f'Number of unique words: {len(word_dist)}')
# find the words which appears only once
rare_words = [word for word, count in word_dist.items() if count == 1]
# remove words appearing only once.
df_overview['plots_without_stopwords'] = df_overview['plots_without_stopwords'].apply(
    lambda plot: [word for word in plot if word not in rare_words])
df_overview['plots_without_stopwords'].head()[0:3]


# %%
before_stop_words_total_number_of_words =\
    len([word for sentence in df_overview['lemmatized_plots']
        for word in sentence])
after_stop_words_total_number_of_words =\
    len([word for sentence in df_overview['plots_without_stopwords']
        for word in sentence])
print('We kept {}% of the words in the corpus'.format(
    round(after_stop_words_total_number_of_words/before_stop_words_total_number_of_words, 2) * 100))

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
print('original movie overview:', df_overview['overview'].iloc[0])
print('processes movie overview:', tokens[0])


# %% [markdown]
# #### Hyperparameters

# %%
def train_LDA(tokens, n_topics, n_passes, no_below, no_above, n_words=12):
    """ Train an LDA model given some hyperparameters
    Args:
        tokens (list): A list of lists, where each inner list is a list of tokens
        n_topics (int): The number of topics to fit the model to
        n_passes (int): The number of passes to run through the data when fitting the model
        no_below (int): The minimum number of documents a word must be in to be included in the model
        no_above (float): The maximum proportion of documents a word can be in to be included in the model
        n_words (int, optional): The number of words to include in the list of topics. Defaults to 12.

    Returns:
        tuple: A tuple containing the trained LDA model, a list of topics, a list of topic distributions
         for each document, and the corpus used to fit the model.
    """
    np.random.seed(42)
    dictionary, corpus = build_dictionnary_and_corpus(
        tokens, no_below=no_below, no_above=no_above)
    lda_model = create_lda_model(corpus, dictionary, n_topics)
    topics = get_topics(lda_model, n_words)
    topic_distribution = get_topic_distribution(lda_model, corpus)
    return lda_model, topics, topic_distribution, corpus


# %%
# no_below = 60  # minimum number of documents a word must be present in to be kept
no_below = 4  # minimum number of documents a word must be present in to be kept, less than 5 gives bad results
no_above = 0.5  # maximum proportion of documents a word can be present in to be kept
n_topics = 10  # number of topics
n_passes = 10  # almost converged after 5 iterations

# %% [markdown]
# ### Dictionnary & Corpus
# The dictionnary will be the list of unique words, and the corpus a list of movie plots bag of words.

# %% [markdown]
# #### LDA Model

# %%
no_below = 10
no_above = 0.5
n_topics = 12
n_passes = 120
n_words = 15
lda_model, topics, topic_distribution, corpus =\
    train_LDA(tokens, n_topics, n_passes, no_below, no_above, n_words)
for i, topic in enumerate(topics):
    print('Topic {}: {}'.format(i, topic))
# for each movie plot, get its topic distribution (i.e the probability of each topic in descending order)
topic_distributions = get_topic_distribution(lda_model, corpus)
print('\n')
for i in range(1, 3):
    print('Movie plot: {}'.format(df_overview['overview'].iloc[i]))
    print('Topic distribution for the first movie plot: {}'.format(
        topic_distributions[i][0:5]))
    print('\n')


# %% [markdown]
# Based the description of each topic, we could assign the following names to each one:
# - Topic 0: **Criminal Investigation**
# - Topic 1: **Coming of Age**
# - Topic 2: **Espionage**
# - Topic 3: **Police Action**
# - Topic 4: **Military Operations**
# - Topic 5: **Crime and Punishment**
# - Topic 6: **War and Political Conflict**
# - Topic 7: **Mystery and Suspense**
# - Topic 8: **Relationships and Family Dynamics**
# - Topic 9: **High School and Youth**
# - Topic 10: **Romance and Love**
# - Topic 11: **Crime and Gangs**

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
df_overview.head()

# %%
names = ['Criminal Investigation',
         'Coming of Age',
         'Espionage',
         'Police Action',
         'Military Operations',
         'Crime and Punishment',
         'War and Political Conflict',
         'Mystery and Suspense',
         'Relationships and Family Dynamics',
         'High School and Youth',
         'Romance and Love',
         'Crime and Gangs']

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
    go.Bar(name='Netflix', x=list(range(n_topics)), y=netflix_topics_dist, width=bar_width, marker_color='#636EFA',
           opacity=0.8),
    go.Bar(name='Prime', x=list(range(n_topics)), y=prime_topics_dist, width=bar_width, marker_color='#FFA15A',
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


# %% [markdown]
# As we can see, the distribution is almost the same for each platform. This isn't so much surprising as
# we've seen before that they have roughly the same proportion of movies for most of the categories.
# Thus using topics in order to determine which streaming platform has the best rating will not be possible. This is
# because as they have same distribution of topic, we cannot discriminate them on the topics.
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

netflix_color = '#636EFA'
prime_color = '#FFA15A'

fig = make_subplots(rows=1, cols=2,
                    subplot_titles=('sentiment polarity distribution', 'sentiment subjectivity distribution'))
data = [
    go.Histogram(name='Netflix', x=netflix_sentiments_polarity, marker_color=netflix_color, opacity=0.5,
                 histnorm='probability', xbins=dict(start=-1, end=1, size=0.1), legendgroup='Netflix'),
    go.Histogram(name='Prime', x=prime_sentiments_polarity, marker_color=prime_color, opacity=0.5,
                 histnorm='probability', xbins=dict(start=-1, end=1, size=0.1)),
    go.Histogram(name='Netflix', x=netflix_sentiments_subjectivity, marker_color=netflix_color, opacity=0.5,
                 histnorm='probability', xbins=dict(start=0, end=1, size=0.05), showlegend=False,
                 legendgroup='Netflix'),
    go.Histogram(name='Prime', x=prime_sentiments_subjectivity, marker_color=prime_color, opacity=0.5,
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
polarity_mean_median_lines = [
    go.Scatter(x=[netflix_polarity_mean]*2, y=polarity_y_range, name='Mean', legendgroup='Mean',
               mode='lines', line=dict(color=netflix_color, width=width, dash='dot')),
    go.Scatter(x=[netflix_polarity_median]*2, y=polarity_y_range, name='Median', legendgroup='Median',
               mode='lines', line=dict(color=netflix_color, width=width, dash='dashdot')),
    go.Scatter(x=[prime_polarity_mean]*2, y=polarity_y_range, mode='lines', name='Prime Mean',
               line=dict(color=prime_color, width=width, dash='dot'), legendgroup='Mean',
               showlegend=False),
    go.Scatter(x=[prime_polarity_median]*2, y=polarity_y_range, legendgroup='Median', showlegend=False,
               name='Prime Median', mode='lines', line=dict(color=prime_color, width=width, dash='dashdot')),
    go.Scatter(x=[netflix_subjectivity_mean]*2, y=subjectivity_y_range, legendgroup='Mean', name='Netflix Mean',
               showlegend=False, mode='lines', line=dict(color=netflix_color, width=width, dash='dot')),
    go.Scatter(x=[netflix_subjectivity_median]*2, y=subjectivity_y_range, legendgroup='Median', name='Netflix Median',
               showlegend=False, mode='lines', line=dict(color=netflix_color, width=width, dash='dashdot')),
    go.Scatter(x=[prime_subjectivity_mean]*2, y=subjectivity_y_range, legendgroup='Mean', name='Prime Mean',
               showlegend=False, mode='lines', line=dict(color=prime_color, width=width, dash='dot')),
    go.Scatter(x=[prime_subjectivity_median]*2, y=subjectivity_y_range, legendgroup='Median', name='Prime Median',
               showlegend=False, mode='lines', line=dict(color=prime_color, width=width, dash='dashdot')),
]
for i, elem in enumerate(polarity_mean_median_lines):
    fig.add_trace(elem, row=1, col=int(1 + i/4))

fig.update_layout(barmode='overlay')
for i, name in enumerate(['Polarity', 'Subjectivity'], start=1):
    fig.update_xaxes(title_text='{} value'.format(name), row=1, col=i)
    fig.update_yaxes(title_text='probability', row=1, col=i)
fig.show()

# %% [markdown]
# From these graphs, we see that the polarity is a little bit higher than 0 for both streaming services. It would
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
# As the p-value is less than 0.05, it suggests that the difference between the means of the two distributions is
# statistically significant, and we can reject the null hypothesis that the two distributions are similar.
# We can see that this analysis is true for both polarity and subjectivity.
#
# We will then see in the following of this project, if polarity and subjectivity of overviews can help to determine
# which streaming service is the best.

# %%
# print the first element of df_overview
print(df_overview['overview'].iloc[0])
print(df_overview['on_prime'].iloc[0])
print(df_overview['on_netflix'].iloc[0])
print(netflix_sentiments[0])
print(prime_sentiments[0])


# %%
# add the prime_sentiments_polarity and netflix_sentiments_polarity to the dataframe
df_overview_prime['sentiments_polarity'] = prime_sentiments_polarity
df_overview_netflix['sentiments_polarity'] = netflix_sentiments_polarity
# add the prime_sentiments_subjectivity and netflix_sentiments_subjectivity to the dataframe
df_overview_prime['sentiments_subjectivity'] = prime_sentiments_subjectivity
df_overview_netflix['sentiments_subjectivity'] = netflix_sentiments_subjectivity
# for each dataframe, remove the tokenized plots, plots_with_POS_tag, lemmatized_plots, plots_without_stopwords
# columns
df_overview_prime = df_overview_prime.drop(
    ['tokenized_plots', 'plots_with_POS_tag', 'lemmatized_plots', 'plots_without_stopwords'], axis=1)
df_overview_netflix = df_overview_netflix.drop(
    ['tokenized_plots', 'plots_with_POS_tag', 'lemmatized_plots', 'plots_without_stopwords'], axis=1)
# concatenate the two dataframes
df_overview_bis = pd.concat([df_overview_prime, df_overview_netflix])
df_overview_bis.head()

# %% [markdown]
# # Observational Study

# %%
df = helper.prepare_df()
df['genres'] = df['genres'].apply(lambda x: x.split(','))
df.reset_index(drop=True, inplace=True)
# merge the original dataframe and the one containing the sentiment analysis
merged_df = df.merge(df_overview_bis, left_index=True,
                     right_index=True, suffixes=('', '_df2'))
# Get the list of common columns between the two dataframes
common_columns = list(set(df.columns) & set(df_overview_bis.columns))

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
def plot_rating_distribution(df: pd.DataFrame):
    """
    Plot the rating distribution on Netflix and Prime using Plotly.

    Args:
        df (pd.DataFrame): The dataframe containing the data,
            must contain the columns 'on_netflix', 'on_prime' and 'averageRating'
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
            marker_color=netflix_color,
        ),
        go.Histogram(
            x=df[df['on_prime']]['averageRating'],
            histnorm='probability',
            xbins=dict(start=0, end=10, size=0.1),
            opacity=0.5,
            name='Prime',
            marker_color=prime_color
        ),
        go.Scatter(
            x=[df[df['on_netflix']]['averageRating'].mean()] * 2,
            y=[0, 1],
            yaxis='y2',
            name='Netflix mean',
            mode='lines',
            line=dict(color=netflix_color, width=1, dash='dash'),
            legendgroup='Mean',
            showlegend=False,
        ),
        go.Scatter(
            x=[df[df['on_netflix']]['averageRating'].median()] * 2,
            y=[0, 1],
            yaxis='y2',
            name='Netflix median',
            mode='lines',
            line=dict(color=netflix_color, width=1, dash='dot'),
            legendgroup='Median',
            showlegend=False,
        ),
        go.Scatter(
            x=[df[df['on_prime']]['averageRating'].mean()] * 2,
            y=[0, 1],
            yaxis='y2',
            name='Prime mean',
            mode='lines',
            line=dict(color=prime_color, width=1, dash='dash'),
            legendgroup='Mean',
            showlegend=False,
        ),
        go.Scatter(
            x=[df[df['on_prime']]['averageRating'].median()] * 2,
            y=[0, 1],
            yaxis='y2',
            name='Prime median',
            mode='lines',
            line=dict(color=prime_color, width=1, dash='dot'),
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

    # show the plots
    fig.show()


plot_rating_distribution(df)


# %%
df[df['on_prime']]['averageRating'].median()

# %% [markdown]
# From this first graph, it seems that you are more likely able to find an high rated movie on Netlfix
# than on Amazon Prime.

# %% [markdown]
# # Matching based on propensity score
# First we'll create an adapted dataframe with only the variables we want to use for the matching.
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
sns.pairplot(df[['averageRating', 'numVotes', 'directors',
                 'release_year', 'runtimeMinutes', 'streaming_service', 'sentiments_polarity']],
             hue='streaming_service')
plt.show()

# %%
df_netflix = matching_df[matching_df['on_netflix'] == 1].copy()
df_netflix.drop(columns=['on_netflix', 'on_prime'], inplace=True)

df_prime = matching_df[matching_df['on_prime'] == 1].copy()
df_prime.drop(columns=['on_netflix', 'on_prime'], inplace=True)


# %%

def plot_hist_matching(df_netflix: pd.DataFrame, df_prime: pd.DataFrame):
    """
    Plot the histogram of the matching features on Netflix and Prime

    Args:
        df_netflix (pd.DataFrame): The dataframe containing the data for Netflix
        df_prime (pd.DataFrame): The dataframe containing the data for Prime
    """

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    for i, col in enumerate(['numVotes', 'release_year', 'runtimeMinutes', 'sentiments_polarity']):
        if col in ['release_year', 'runtimeMinutes', 'sentiments_polarity']:
            axs[i//2, i % 2].hist(df_netflix[col], alpha=0.5,
                                  label='Netflix', density=True, bins=20)
            axs[i//2, i % 2].hist(df_prime[col], alpha=0.5,
                                  label='Prime', density=True, bins=20)
            axs[i//2, i % 2].set_ylabel('density')
        elif col in ['numVotes']:
            max_x_value = max(df_netflix[col].max(), df_prime[col].max())
            bins_logspace = np.logspace(0, np.log10(max_x_value), 40)

            axs[i//2, i % 2].hist(df_netflix[col], alpha=0.5,
                                  label='Netflix', bins=bins_logspace)
            axs[i//2, i % 2].hist(df_prime[col], alpha=0.5,
                                  label='Prime', bins=bins_logspace)
            axs[i//2, i % 2].set_xscale('log')
            axs[i//2, i % 2].set_ylabel('number of movies')
        else:
            # should never happen
            raise ValueError('column not found')
        axs[i//2, i % 2].legend()
        axs[i//2, i % 2].set_title(col)
    fig.tight_layout()
    plt.show()


plot_hist_matching(df_netflix, df_prime)
# compute some statistics to see if the distributions are different
for col in ['numVotes', 'runtimeMinutes', 'sentiments_polarity', 'release_year']:
    t_statistic_polarity, p_value_polarity = ttest_ind(
        df_prime[col], df_netflix[col], equal_var=False)
    print(f'{col}: t-statistic: {round(t_statistic_polarity, 2)}, p-value: {round(p_value_polarity,2)}')


# %%

def plot_hist_matching(df_netflix: pd.DataFrame, df_prime: pd.DataFrame):
    """
    Plot the histogram of the matching features on Netflix and Prime

    Args:
        df_netflix (pd.DataFrame): The dataframe containing the data for Netflix
        df_prime (pd.DataFrame): The dataframe containing the data for Prime
    """

    columns_to_plot = ['numVotes', 'release_year',
                       'runtimeMinutes', 'sentiments_polarity']

    fig = make_subplots(rows=2, cols=2, subplot_titles=columns_to_plot)

    for i, col in enumerate(columns_to_plot):
        if col in ['release_year', 'runtimeMinutes', 'sentiments_polarity']:
            histograms = [
                go.Histogram(
                    x=df_netflix[col],
                    name='Netflix',
                    histnorm='probability',
                    legendgroup='Netflix',
                    marker_color=netflix_color,
                    bingroup=col,
                    showlegend=False,
                ),
                go.Histogram(
                    x=df_prime[col],
                    name='Prime',
                    histnorm='probability',
                    legendgroup='Prime',
                    marker_color=prime_color,
                    bingroup=col,
                    showlegend=False,
                ),
            ]

            for histogram in histograms:
                fig.add_trace(histogram, row=i//2+1, col=i % 2+1)

        elif col in ['numVotes']:
            # max_x_value = max(df_netflix[col].max(), df_prime[col].max())
            # bins_logspace = np.logspace(0, np.log10(max_x_value), 40)

            histograms = [
                go.Histogram(
                    x=df_netflix[col],
                    name='Netflix',
                    histnorm='probability',
                    legendgroup='Netflix',
                    marker_color=netflix_color,
                    xaxis='x2',
                    xbins=dict(start=0, end=10, size=0.1),
                    # transforms=[dict(type='log')]
                ),

                go.Histogram(
                    x=df_prime[col],
                    name='Prime',
                    histnorm='probability',
                    legendgroup='Netflix',
                    marker_color=prime_color,
                ),
            ]

            for histogram in histograms:
                fig.add_trace(histogram, row=i//2+1, col=i % 2+1)

            # fig.update_xaxes(type='log', row=i//2+1, col=i % 2+1)
        else:
            # should never happen
            raise ValueError('column not found')

    fig.update_layout(barmode='overlay')
    fig.update_traces(opacity=0.5)
    fig.show()


plot_hist_matching(df_netflix, df_prime)
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
def plot_genre_distribution(df_netflix: pd.DataFrame, df_prime: pd.DataFrame):
    """Plot the genre distribution on Netflix and Prime

    Args:
        df_netflix (pd.DataFrame): The dataframe containing the data for Netflix
        df_prime (pd.DataFrame): The dataframe containing the data for Prime
    """

    # genres should be sorted by occurences
    max_y_axis = max(df_netflix[genres].sum().max(
    ) / len(df_netflix), df_prime[genres].sum().max() / len(df_prime))
    # round up to next multiple of 0.1
    max_y_axis = math.ceil(max_y_axis * 10) / 10

    fig, ax = plt.subplots(figsize=(15, 5))
    bar_width = 0.35
    # genres = ['Horror', 'Action']

    # we add the bar_width to the x axis to have the bars side by side
    ax.bar(np.arange(len(genres)) + bar_width, df_netflix[genres].sum() / len(df_netflix), bar_width,
           alpha=0.5, label='Netflix')
    ax.bar(np.arange(len(genres)), df_prime[genres].sum() / len(df_prime), bar_width,
           alpha=0.5, label='Prime')

    # set the x axis ticks to the genre names
    ax.set_xticklabels(genres, rotation=90)
    ax.set_xticks(np.arange(len(genres)) + bar_width / 2)

    ax.set_xlabel('Genre')
    ax.set_ylabel('% of Movies')
    ax.set_title('% of Movies per Genre per Streaming Service')
    ax.legend()

    plt.show()


plot_genre_distribution(df_netflix, df_prime)

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
model = smf.logit(formula='on_netflix ~ normalized_numVotes + normalized_release_year + '
                  'normalized_runtimeMinutes + sentiments_polarity',
                  data=matching_df)

res = model.fit()
matching_df['predicted_netflix'] = res.predict(matching_df)

print(res.summary())


# %%
# generate a report on the classification
print(classification_report(matching_df['on_netflix'], matching_df['predicted_netflix'].apply(
    lambda x: 1 if x > 0.5 else 0)))


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
# only keep 20%
df_netflix = df_netflix.sample(frac=0.2)
df_prime = df_prime.sample(frac=0.2)


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
plot_hist_matching(df_netflix, df_prime)
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
plot_genre_distribution(df_netflix, df_prime)

# %%
plot_rating_distribution(balanced_df)

# %%
t_statistic_rating, p_value_rating = ttest_ind(
    balanced_df[balanced_df['on_prime']
                ]['averageRating'], balanced_df[balanced_df['on_netflix']]['averageRating'],
    equal_var=False)
print(
    f'averageRating: t-statistic: {round(t_statistic_rating, 2)}, p-value: {round(p_value_rating,2)}')

# %% [markdown]
# We now redo the same analysis as before, but this time we'll do a perfect matching also on the production country.

# %%
netflix_row

# %%
mask_netflix_only = (matching_df['on_netflix'] == 1) & (
    matching_df['on_prime'] == 0)
mask_prime_only = (matching_df['on_netflix'] == 0) & (
    matching_df['on_prime'] == 1)

df_netflix = matching_df[mask_netflix_only].copy()
df_netflix.drop(columns=['on_netflix', 'on_prime'], inplace=True)

df_prime = matching_df[mask_prime_only].copy()
df_prime.drop(columns=['on_netflix', 'on_prime'], inplace=True)

# only keep 20%
df_netflix = df_netflix.sample(frac=0.2)
df_prime = df_prime.sample(frac=0.2)

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
plot_hist_matching(df_netflix, df_prime)
for col in ['numVotes', 'runtimeMinutes', 'sentiments_polarity', 'release_year']:
    t_statistic_polarity, p_value_polarity = ttest_ind(
        df_prime[col], df_netflix[col], equal_var=False)
    print(f'{col}: t-statistic: {round(t_statistic_polarity, 2)}, p-value: {round(p_value_polarity,2)}')

# %%
plot_rating_distribution(balanced_df)

# %%
t_statistic_rating, p_value_rating = ttest_ind(
    balanced_df[balanced_df['on_prime']
                ]['averageRating'], balanced_df[balanced_df['on_netflix']]['averageRating'],
    equal_var=False)
print(
    f'averageRating: t-statistic: {round(t_statistic_rating, 2)}, p-value: {round(p_value_rating,2)}')

# %%
