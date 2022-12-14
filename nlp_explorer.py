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
from textblob import TextBlob
import pyLDAvis.gensim_models
from gensim.models import LdaMulticore
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
# take a sample of 10% of the movies
# df_overview = df_overview_complete.sample(frac=0.1, random_state=42)
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
print(tokens[0:2])


# %% [markdown]
# #### Hyperparameters

# %%
def train_LDA(tokens, n_topics, n_passes, no_below, no_above, n_words=12):
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
# BEST ONE
no_below = 3
no_above = 0.5
n_topics = 11
n_passes = 10
lda_model, topics, topic_distribution, corpus =\
    train_LDA(tokens, n_topics, n_passes, no_below, no_above)
# get the topics
topics = get_topics(lda_model, num_topics=n_topics, num_words=15)
# print topics with new line
for i, topic in enumerate(topics):
    print('Topic {}: {}'.format(i, topic))
# for each movie plot, get its topic distribution (i.e the probability of each topic) in descending order
topic_distributions = get_topic_distribution(lda_model, corpus)
print('\n')
for i in range(0, 10):
    print('Movie plot: {}'.format(df_overview['overview'].iloc[i]))
    print('Topic distribution for the first movie plot: {}'.format(
        topic_distributions[i][0:5]))
    print('\n')

# %%
print(len(topic_distributions))
# get the movies only on netflix
df_overview_netflix = df_overview[df_overview['on_netflix'] == 1]
# get the movies only on prime
df_overview_prime = df_overview[df_overview['on_prime'] == 1]

# get the index of each movie on netflix
netflix_index = df_overview_netflix.index.tolist()
# get the topic distribution for each movie on netflix
netflix_topic_distribution = [topic_distributions[i] for i in netflix_index]

netflix_topics_dist = [0] * n_topics
for topic_dist in netflix_topic_distribution:
    for i in range(n_topics):
        netflix_topics_dist[topic_dist[i][0]] += topic_dist[i][1]

# divide by the total number of movies on netflix
netflix_topics_dist = [x / len(df_overview_netflix)
                       for x in netflix_topics_dist]
print(len(netflix_index))

prime_index = df_overview_prime.index.tolist()
# get the topic distribution for each movie on netflix
prime_topic_distribution = [topic_distributions[i] for i in prime_index]
print(len(prime_index))


prime_topics_dist = [0] * n_topics
for topic_dist in prime_topic_distribution:
    for i in range(n_topics):
        prime_topics_dist[topic_dist[i][0]] += topic_dist[i][1]

# divide by the total number of movies on netflix
prime_topics_dist = [x / len(df_overview_prime) for x in prime_topics_dist]
print(prime_topics_dist)
print(netflix_topics_dist)


# plot the two distributions
plt.figure(figsize=(10, 5))
plt.bar(range(n_topics), prime_topics_dist, label='Prime', alpha=1)
plt.bar(range(n_topics), netflix_topics_dist, label='Netflix', alpha=0.5)
plt.legend()
plt.show()

print(sum(prime_topics_dist))
print(sum(netflix_topics_dist))


# %%
# models
no_below = 5
no_above = 0.5
dictionary, corpus = build_dictionnary_and_corpus(
    tokens, no_below=no_below, no_above=no_above)
print('Dictionary size: {}'.format(len(dictionary)))
print('Dictionary first 10 elements: {}'.format(
    list(dictionary.items())[0:10]))
print('Corpus size: {}'.format(len(corpus)))
print('Corpus first 2 elements: {}'.format(corpus[0:2]))

params = {'passes': 10, 'random_state': 42}
base_models = dict()
model = LdaMulticore(corpus=corpus, num_topics=4, id2word=dictionary, workers=6,
                     passes=params['passes'], random_state=params['random_state'])

# %%
model.show_topics(num_words=5)

# %%
model.show_topic(1, 20)

# %%
# plot topics
data = pyLDAvis.gensim_models.prepare(model, corpus, dictionary)
pyLDAvis.display(data)

# %%
# predict the topic distribution for the first movie plot
model[corpus[0]]

# %%
temp = df_overview.copy()
df_prime = df_overview[df_overview['on_prime'] == 1]
df_netflix = df_overview[df_overview['on_netflix'] == 1]
# print the number of movies on each platform
print('Number of movies on Prime: {}'.format(len(df_prime)))
print('Number of movies on Netflix: {}'.format(len(df_netflix)))

# %%

# Create a TextBlob object for each movie plot
netflix_blobs = [TextBlob(' '.join(plot))
                 for plot in df_netflix['plots_without_stopwords']]
prime_blobs = [TextBlob(' '.join(plot))
               for plot in df_prime['plots_without_stopwords']]

# Use the `sentiment` property to get the overall sentiment of each plot
netflix_sentiments = [blob.sentiment for blob in netflix_blobs]
prime_sentiments = [blob.sentiment for blob in prime_blobs]

# # plot the distribution of the polarity and subjectivity. Normalized to 1
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 5))
# plt.hist([sentiment.polarity for sentiment in prime_sentiments], bins=20, label='Prime', alpha=0.5)
# plt.hist([sentiment.polarity for sentiment in netflix_sentiments], bins=20, label='Netflix', alpha=0.5)
# plt.legend()
# plt.show()

# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 5))
# plt.hist([sentiment.subjectivity for sentiment in prime_sentiments], bins=20, label='Prime', alpha=0.5)
# plt.hist([sentiment.subjectivity for sentiment in netflix_sentiments], bins=20, label='Netflix', alpha=0.5)
# plt.legend()
# plt.show()

# divide each bins by the total number of movies on each platform
prime_polarity = [0] * 20
netflix_polarity = [0] * 20
for sentiment in prime_sentiments:
    prime_polarity[int((sentiment.polarity+1) * 10) - 1] += 1
for sentiment in netflix_sentiments:
    netflix_polarity[int((sentiment.polarity+1) * 10) - 1] += 1

prime_polarity = [x / len(df_prime) for x in prime_polarity]
netflix_polarity = [x / len(df_netflix) for x in netflix_polarity]
print(prime_polarity)
print(netflix_polarity)
print(sum(prime_polarity))
print(sum(netflix_polarity))
# plot prime polarity
plt.figure(figsize=(10, 5))
# plot on a range from -5 to 5, with 20 values
plt.plot(range(-10, 10), prime_polarity, label='Prime')
plt.plot(range(-10, 10), netflix_polarity, label='Netflix')
plt.xlabel(xlabel='Polarity')
plt.legend()
plt.show()

# prime_polarity = [0] * 20
# netflix_polarity = [0] * 20
# for sentiment in prime_sentiments:
#     prime_polarity[int((sentiment.subjectivity+1) * 10) - 1] += 1
# for sentiment in netflix_sentiments:
#     netflix_polarity[int((sentiment.subjectivity+1) * 10) - 1] += 1

# prime_polarity = [x / len(df_prime) for x in prime_polarity]
# netflix_polarity = [x / len(df_netflix) for x in netflix_polarity]
# print(prime_polarity)
# print(netflix_polarity)
# print(sum(prime_polarity))
# print(sum(netflix_polarity))
# # plot prime polarity
# plt.figure(figsize=(10, 5))
# plt.bar(range(20), prime_polarity, label='Prime')
# plt.bar(range(20), netflix_polarity, label='Netflix')
# plt.legend()
# plt.show()


# %%
