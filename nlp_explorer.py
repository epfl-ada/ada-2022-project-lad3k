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
from tqdm import tqdm
import networkx as nx
from sklearn.metrics import classification_report
import statsmodels.formula.api as smf
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import seaborn as sns
from src import helper
from scipy.stats import ttest_ind
from textblob import TextBlob
import matplotlib.pyplot as plt
import nltk
import string
import numpy as np
from nltk import pos_tag
from gensim.models import Phrases
from nltk.corpus import stopwords
from src.helper import prepare_df
from src.nlp_helper import get_wordnet_pos, build_dictionnary_and_corpus,\
    create_lda_model, get_topics, get_topic_distribution
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

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
# not so bad
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


# plot the two distributions
plt.figure(figsize=(10, 5))
bar_width = 0.4
plt.bar(np.array(range(n_topics)) + bar_width, netflix_topics_dist,
        label='Netflix', alpha=0.5, width=bar_width)
plt.bar(range(n_topics), prime_topics_dist,
        label='Prime', alpha=0.5, width=bar_width)
# center tick marks and labels on middle of the two columns
tick_positions = np.array(range(n_topics)) + bar_width / 2
plt.xticks(tick_positions, range(n_topics))
plt.xticks(tick_positions, names, rotation=90)
plt.xlabel('Topic')
plt.ylabel('Distribution over topics')
plt.title('Topic distribution for movies on Netflix and Prime')
plt.legend()
plt.show()


# %% [markdown]
# As we can see, the distribution is almost the same for each platform. This isn't so much surprising as
# we've seen before that they have roughly the same proportion of movies for most of the categories.
# Thus using topics in order to determine which streaming platform has the best rating will not be possible. This is
# because as they have same distribution of topicc, we cannot discriminate them on the topics.
#
#
# ### NLP with sentiment analysis
#
# We can explore if movies overviews "sentiments" (positive/negative) are different between the two streaming platforms.
# Polarity is a measure of the sentiment of the overview, with values ranging from -1 (very negative) to
# 1 (very positive). Subjectivity is a measure of the objectivity of the overview, with values
# ranging from 0 (completely objective) to 1 (completely subjective).

# %%
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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

counts_netflix_polarity, bins, patches = ax1.hist(
    netflix_sentiments_polarity, bins=20, alpha=0.5, label='Netflix', density=True)
counts_prime_polarity, bins, patches = ax1.hist(
    prime_sentiments_polarity, bins=20, alpha=0.5, label='Prime', density=True)

ax1.axvline(np.mean(netflix_sentiments_polarity), color='b',
            linestyle='dashed', linewidth=1, label='Netflix mean')
ax1.axvline(np.mean(prime_sentiments_polarity), color='r',
            linestyle='dashed', linewidth=1, label='Prime mean')

counts_netflix_subjectivity, bins, patches = ax2.hist(
    netflix_sentiments_subjectivity, bins=20, alpha=0.5, label='Netflix', density=True)
counts, bins, patches = ax2.hist(
    prime_sentiments_subjectivity, bins=20, alpha=0.5, label='Prime', density=True)

ax2.axvline(np.mean(netflix_sentiments_subjectivity), color='b',
            linestyle='dashed', linewidth=1, label='Netflix mean')
ax2.axvline(np.mean(prime_sentiments_subjectivity), color='r',
            linestyle='dashed', linewidth=1, label='Prime mean')

ax1.set_title('Polarity Distribution')
ax1.set_xlabel('Polarity')
ax1.set_ylabel('Percentage of Movies (divided by 10)')
ax2.set_title('Subjectivity Distribution')
ax2.set_xlabel('Subjectivity')
ax2.set_ylabel('Percentage of Movies (divided by 10)')
ax1.legend()
ax2.legend()
plt.show()


# %% [markdown]
# From these graphs, we see that the polarity is a little bit higher than 0 for both streaming services. It would
# also seems that Prime have movies with a lower polarity than Netflix, i.e. Prime movies overviews would have a
# more "negative" sentiment. For subjectivity, we see both distributions have roughly the same distribution. We also see
# that Prime seems to have more "objective" movies overview than Neflix. Netflix on its side seems to have a little bit
# more "subjective" overviews.

# %%

# Perform a t-test to compare the means of the polarity distributions
t_statistic_polarity, p_value_polarity = ttest_ind(
    prime_sentiments_polarity, netflix_sentiments_polarity)
t_statistic_subjectivity, p_value_subjectivity = ttest_ind(
    prime_sentiments_subjectivity, netflix_sentiments_subjectivity)

print('Polarity p-value', p_value_polarity)
print('Subjectivity p-value', p_value_subjectivity)


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

# %%
# Create the dataframe from our datasets
df = helper.prepare_df()
df['genres'] = df['genres'].apply(lambda x: x.split(','))

# %%
print(len(df_overview_bis))
print(len(df))
df.head()
df_new = df.reset_index(drop=True)
# print the index of the first movie in df_overview_bis
print(df_overview_bis.index[0])
#  print the index of the first movie in df
print(df_new.index[0])
# print the df at index 7496
print(df_new.loc[7496])
# merge the 2 dataframes on the index
df_overview_bis = df_overview_bis.reset_index(drop=True)
df_new = df_new.reset_index(drop=True)
df_overview_bis = df_overview_bis.merge(
    df_new, left_index=True, right_index=True)
df_overview_bis.head()

# %% [markdown]
# As the p-value is less than 0.05, it suggests that the difference between the means of the two distributions is
# statistically significant, and we can reject the null hypothesis that the two distributions are similar.
# We can see that this analysis is true for both polarity and subjectivity.
#
# We will then see in the following of this project, if polarity and subjectivity of overviews can help to determine
# which streaming service is the best.

# %% [markdown]
# # Observational Studies

# %%
# Create the dataframe from our datasets
df = helper.prepare_df()
df['genres'] = df['genres'].apply(lambda x: x.split(','))
df.head()


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
    Plot the rating distribution on Netflix and Prime

    Args:
        df (pd.DataFrame): The dataframe containing the data,
            must contain the columns 'on_netflix', 'on_prime' and 'averageRating'
    """
    # if needed transform binary 'on_netflix' and 'on_prime' to boolean
    df['on_netflix'] = df['on_netflix'].astype(bool)
    df['on_prime'] = df['on_prime'].astype(bool)

    # plot the movie rating distribution on Netflix and Prime
    # rating is in column "averageRating"
    # a col "streaming_service" tells us if the movie is on Netflix, Prime or both
    plt.hist(df[df['on_netflix']]['averageRating'],
             bins=np.arange(0, 10.1, 0.5),
             alpha=0.5,
             density=True,
             color='C0',
             label='Netflix')
    plt.axvline(df[df['on_netflix']]['averageRating'].mean(),
                color='C0', linestyle='dashed', linewidth=1)
    plt.axvline(df[df['on_netflix']]['averageRating'].median(),
                color='C0', linestyle='dotted', linewidth=1)

    plt.hist(df[df['on_prime']]['averageRating'],
             bins=np.arange(0, 10.1, 0.5),
             alpha=0.5,
             density=True,
             color='C1',
             label='Prime')
    plt.axvline(df[df['on_prime']]['averageRating'].mean(),
                color='C1', linestyle='dashed', linewidth=1)
    plt.axvline(df[df['on_prime']]['averageRating'].median(),
                color='C1', linestyle='dotted', linewidth=1)

    # add legend for hist and mean/median
    legend_elements = [
        plt.Line2D([0], [0], color='C0', lw=4, label='Netflix'),
        plt.Line2D([0], [0], color='C1', lw=4, label='Prime'),
        plt.Line2D([0], [0], color='k', lw=1,
                   linestyle='dashed', label='Mean'),
        plt.Line2D([0], [0], color='k', lw=1,
                   linestyle='dotted', label='Median')
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.title('Movie rating distribution on Netflix and Prime')
    plt.xlabel('Average rating')
    plt.ylabel('Density')
    plt.show()


plot_rating_distribution(df)


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
                  'runtimeMinutes', 'genres', 'on_netflix', 'on_prime']].copy()
# we only keep numbers or values that we'll be able to transform to binary

matching_df['on_netflix'] = matching_df['on_netflix'].apply(
    lambda x: 1 if x else 0)
matching_df['on_prime'] = matching_df['on_prime'].apply(
    lambda x: 1 if x else 0)

# %%
# create a list of genres with their occurences
genres = []
for genre in matching_df['genres']:
    genres.extend(genre)
genres = pd.Series(genres).value_counts()
top_5_genres = genres[:5].index
print(f'Top 5 genres: {top_5_genres.values}')

# compute in how many movies there is at least one of the top 5 genres
top_5_genres_df = matching_df[matching_df['genres'].apply(
    lambda x: any(genre in x for genre in top_5_genres))]
print(f'There is {top_5_genres_df.shape[0]/matching_df.shape[0]*100:.2f}'
      '% of movies with at least one of the top 5 genres')


# %%
mlb = MultiLabelBinarizer(classes=top_5_genres)

genres_df = pd.DataFrame(mlb.fit_transform(
    matching_df['genres']), columns=mlb.classes_, index=matching_df.index)
matching_df = pd.concat([matching_df, genres_df], axis=1)
matching_df.drop(columns=['genres'], inplace=True)
matching_df.head()

# %%
# pairplot but only with averageRating
sns.pairplot(df[['averageRating', 'numVotes', 'directors',
                 'release_year', 'runtimeMinutes', 'streaming_service']], hue='streaming_service')
plt.show()

# %%
df_netflix = matching_df[matching_df['on_netflix'] == 1].copy()
df_netflix.drop(columns=['on_netflix', 'on_prime'], inplace=True)

df_prime = matching_df[matching_df['on_prime'] == 1].copy()
df_prime.drop(columns=['on_netflix', 'on_prime'], inplace=True)


# %%
fig, axs = plt.subplots(3, 3, figsize=(15, 15))
for i, col in enumerate([x for x in df_netflix.columns if x not in ['averageRating']]):
    if col in ['release_year', 'runtimeMinutes']:
        axs[i // 3, i % 3].hist(df_netflix[col], alpha=0.5,
                                label='Netflix', density=True, bins=20)
        axs[i // 3, i % 3].hist(df_prime[col], alpha=0.5,
                                label='Prime', density=True, bins=20)
        axs[i // 3, i % 3].set_ylabel('density')
    elif col in ['numVotes']:

        max_x_value = max(df_netflix[col].max(), df_prime[col].max())
        bins_logspace = np.logspace(0, np.log10(max_x_value), 40)

        axs[i // 3, i % 3].hist(df_netflix[col], alpha=0.5,
                                label='Netflix', bins=bins_logspace)
        axs[i // 3, i % 3].hist(df_prime[col], alpha=0.5,
                                label='Prime', bins=bins_logspace)
        axs[i // 3, i % 3].set_xscale('log')
        axs[i // 3, i % 3].set_ylabel('number of movies')
    else:
        # binary features, columns charts is the more appropriate
        width = 0.25
        x = np.arange(2)
        axs[i // 3, i % 3].bar(x + width/2, df_netflix[col].value_counts(
            normalize=True), width=width, label='Netflix', alpha=0.5)
        axs[i // 3, i % 3].bar(x - width/2, df_prime[col].value_counts(
            normalize=True), width=width, label='Prime', alpha=0.5)
        axs[i // 3, i % 3].set_xticks(x, ('0', '1'))
        axs[i // 3, i % 3].set_ylabel('density')
    axs[i // 3, i % 3].legend()
    axs[i // 3, i % 3].set_title(col)
axs[-1, -1].axis('off')  # hide last subplot as nothing in it
plt.show()


# %% [markdown]
# We'll now compute a propensity score for each observation using a logistic regression.

# %%
features_to_normalize = ['numVotes', 'release_year', 'runtimeMinutes']
for feature in features_to_normalize:
    matching_df['normalized_' + feature] = (
        matching_df[feature] - matching_df[feature].mean()) / matching_df[feature].std()

# %%
matching_df.columns

# %%
model = smf.logit(formula='on_netflix ~ normalized_numVotes + normalized_release_year + '
                  'normalized_runtimeMinutes',
                  data=matching_df)
# + C(Drama) + C(Comedy) + C(Action) + C(Romance) + C(Thriller)

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
# only keep 10%
df_netflix = df_netflix.sample(frac=0.1)
df_prime = df_prime.sample(frac=0.1)


# %%
G = nx.Graph()

# Loop through all the pairs of instances
for netflix_id, netflix_row in tqdm(df_netflix.iterrows(), total=df_netflix.shape[0]):
    for prime_id, prime_row in df_prime.iterrows():

        # (less edges in the graph, faster computation)
        # here we put some conditions to avoid adding edges between instances that are too different

        # Calculate the similarity
        similarity = get_similarity(netflix_row['predicted_netflix'],
                                    prime_row['predicted_netflix'])

        # we want a similarity of at least 0.5
        if similarity < 0.5:
            continue

        # if genres are different, skip
        if any(netflix_row[top_5_genres] != prime_row[top_5_genres]):
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
plot_rating_distribution(balanced_df)

# %%
df_netflix.columns

# %%
fig, axs = plt.subplots(3, 3, figsize=(15, 15))
for i, col in enumerate(
    [x for x in df_netflix.columns if x not in ['averageRating', 'predicted_netflix', 'normalized_numVotes',
                                                'normalized_release_year', 'normalized_runtimeMinutes']]):
    if col in ['release_year', 'runtimeMinutes']:
        axs[i // 3, i % 3].hist(df_netflix[col], alpha=0.5,
                                label='Netflix', density=True, bins=20)
        axs[i // 3, i % 3].hist(df_prime[col], alpha=0.5,
                                label='Prime', density=True, bins=20)
        axs[i // 3, i % 3].set_ylabel('density')
    elif col in ['numVotes']:

        max_x_value = max(df_netflix[col].max(), df_prime[col].max())
        bins_logspace = np.logspace(0, np.log10(max_x_value), 40)

        axs[i // 3, i % 3].hist(df_netflix[col], alpha=0.5,
                                label='Netflix', bins=bins_logspace)
        axs[i // 3, i % 3].hist(df_prime[col], alpha=0.5,
                                label='Prime', bins=bins_logspace)
        axs[i // 3, i % 3].set_xscale('log')
        axs[i // 3, i % 3].set_ylabel('number of movies')
    else:
        # binary features, columns charts is the more appropriate
        width = 0.25
        x = np.arange(2)
        axs[i // 3, i % 3].bar(x + width/2, df_netflix[col].value_counts(
            normalize=True), width=width, label='Netflix', alpha=0.5)
        axs[i // 3, i % 3].bar(x - width/2, df_prime[col].value_counts(
            normalize=True), width=width, label='Prime', alpha=0.5)
        axs[i // 3, i % 3].set_xticks(x, ('0', '1'))
        axs[i // 3, i % 3].set_ylabel('density')
    axs[i // 3, i % 3].legend()
    axs[i // 3, i % 3].set_title(col)
axs[-1, -1].axis('off')  # hide last subplot as nothing in it
plt.show()

# %%
