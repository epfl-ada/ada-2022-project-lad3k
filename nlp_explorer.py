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
plt.bar(range(n_topics), prime_topics_dist,
        label='Prime', alpha=1, width=bar_width)
plt.bar(np.array(range(n_topics)) + bar_width, netflix_topics_dist,
        label='Netflix', alpha=0.5, width=bar_width)
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


# %% [markdown]
# As the p-value is less than 0.05, it suggests that the difference between the means of the two distributions is
# statistically significant, and we can reject the null hypothesis that the two distributions are similar.
# We can see that this analysis is true for both polarity and subjectivity.
#
# We will then see in the following of this project, if polarity and subjectivity of overviews can help to determine
# which streaming service is the best.

# %%
