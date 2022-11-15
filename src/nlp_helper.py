# import numpy as np
from gensim import models
# from gensim import corpora
# # to flatten list of sentences of tokens into list of tokens
# from itertools import chain
# from nltk.corpus import stopwords
# import nltk
# from nltk import pos_tag
# from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from gensim import corpora
# import pandas as pd
# from nltk.tokenize import word_tokenize
import pandas as pd
# from gensim.models import Phrases


# df = pd.read_csv("../data/moviedb_data csv.gz", compression="gzip")


def read_moviedb_data():
    """
    Read data from moviedb
    :return: dataframe
    """
    df = pd.read_csv('../data/moviedb_data.csv.gz', compression='gzip')
    return df


# # keep only the overview and providers columns
# df_overview_provider = df[["overview", "providers"]]

# df_overview_provider.head()
# # replace nan for provider by {}
# df_overview_provider["providers"] = df_overview_provider["providers"].fillna(
#     "{}")
# df_overview_provider["overview"] = df_overview_provider["overview"].fillna("")
# # copy the dataframe
# df_plots = df_overview_provider.copy()

# df_plots['tokens_sentences'] = df_plots['overview'].apply(
#     lambda movie_plot: word_tokenize(movie_plot))

# lemmatize

# Inspired from https://stackoverflow.com/a/15590384


def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''


# lemmatizer = WordNetLemmatizer()

# df_plots['POS_tokens'] = df_plots['tokens_sentences'].apply(
#     lambda tokens_sentences: pos_tag(tokens_sentences))
# print(df_plots['POS_tokens'].head(1))

# nltk.download('omw-1.4')
# df_plots['tokens_sentences_lemmatized'] = df_plots['POS_tokens'].apply(
#     lambda tokenized_plot: [lemmatizer.lemmatize(word[0], get_wordnet_pos(word[1]))
#                             if get_wordnet_pos(word[1]) != '' else word[0] for word in tokenized_plot])
# df_plots['tokens_sentences_lemmatized'].head(1)


# # Remove stop words

# stopwords_verbs = ['say', 'get', 'go', 'know', 'may', 'need', 'like',
#                    'make', 'see', 'want', 'come', 'take', 'use', 'would', 'can']
# stopwords_other = ['one', 'mr', 'bbc', 'image', 'getty',
#                    'de', 'en', 'caption', 'also', 'copyright', 'something']
# my_stopwords = stopwords.words(
#     'English') + stopwords_verbs + stopwords_other + [".", ","]
# df_plots['tokens'] = df_plots['tokens_sentences_lemmatized'].map(lambda tokenized_plot:\
#  [word.lower() for word in tokenized_plot if word.isalpha() and word.lower() not in my_stopwords and len(word) > 1])
# df_plots['tokens'].head()

# # LDA

# tokens = df_plots['tokens'].tolist()
# bigram_model = Phrases(tokens)
# trigram_model = Phrases(bigram_model[tokens], min_count=1)
# tokens = list(trigram_model[bigram_model[tokens]])


def build_dictionnary_and_corpus(tokens, no_below=60, no_above=0.5):
    dictionary = corpora.Dictionary(tokens)
    dictionary.filter_extremes(no_below=10, no_above=0.5)
    corpus = [dictionary.doc2bow(token) for token in tokens]
    return dictionary, corpus


# dictionary_LDA = corpora.Dictionary(tokens)
# dictionary_LDA.filter_extremes(no_below=60)
# # print each word and its id
# for key, value in dictionary_LDA.items():
#     print(key, ' : ', value)
# corpus = [dictionary_LDA.doc2bow(tok) for tok in tokens]
# # print(corpus)


def create_lda_model(corpus, dictionary, num_topics=10, passes=10):
    lda_model = models.LdaModel(corpus=corpus,
                                id2word=dictionary,
                                num_topics=num_topics,
                                passes=passes,
                                random_state=100,
                                update_every=1,
                                chunksize=100,
                                alpha='auto',
                                per_word_topics=True)
    return lda_model


# np.random.seed(9999)
# num_topics = 15
# lda_model = models.LdaModel(corpus, num_topics=num_topics,
#                            id2word=dictionary_LDA,
#                            passes=4, alpha=[0.01]*num_topics,
#                            eta=[0.01]*len(dictionary_LDA.keys()))


def get_topics(lda_model, num_topics=15, num_words=13):
    topics = []
    for _, topic in lda_model.show_topics(formatted=True, num_topics=num_topics, num_words=13):
        topics.append(' '.join([word.split('*')[1]
                      for word in topic.split(' + ')]).replace("\"", ''))
    return topics


# topics = []


def get_topic_distribution(lda_model, corpus):
    topic_distributions = []
    for doc in corpus:
        topic_distributions.append(lda_model.get_document_topics(doc))
    return topic_distributions


# print the topic distribution for each plot
# topic_distributions = get_topic_distribution(lda_model, corpus)


def print_movie_plot_with_topic_words(topic_distributions, topics, df_plots, num_plots):
    # TODO add random sampling
    for i, distribution in enumerate(topic_distributions[:num_plots]):
        print('plot: {}'.format(df_plots.iloc[i]['overview']))
        # get the value of the topic with the highest probability
        best_topic = max(distribution, key=lambda x: x[1])[0]
        print('associated best topic: {}'.format(topics[best_topic]))
        print()


# # for the first 20 plots, print the topic distribution
# for i, distribution in enumerate(topic_distributions[:15]):
#     print("plot: {}".format(df_plots.iloc[i]['overview']))
#     # get the value of the topic with the highest probability
#     best_topic = max(distribution, key=lambda x: x[1])[0]
#     print("associated best topic: {}".format(topics[best_topic]))
#     print()
