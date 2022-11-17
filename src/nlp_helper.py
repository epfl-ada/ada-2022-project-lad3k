from gensim import models
from gensim import corpora
from nltk.corpus import wordnet


def get_wordnet_pos(treebank_tag):
    """Convert the treebank tag to wordnet tag

    I've used code from this post: https://stackoverflow.com/a/15590384

    Args:
        treebank_tag (_type_): treebank tag

    Returns:
        wordnet tag
    """

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


def build_dictionnary_and_corpus(tokens, no_below, no_above):
    """_summary_

    Args:
        tokens (str[][]): list of tokens, i.e list of list of words
        no_below (int, optional): words must occure in at least no_below plots.
        no_above (float, optional): word must not occure in more than no_above % of the plots.

    Returns:
        (gensim.corpora.dictionary.Dictionary, list): Dictionary and corpus (list of bag of words,
         i.e number of occurence of each word in the plot)
    """
    # this will be used to identify the words in the plots (i.e every word has to belong to dictionary)
    dictionary = corpora.Dictionary(tokens)
    # we remove words that appear in less than no_below movies plots and in more than no_above% of the movies plots
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    # for each plot (token), we compute how many times each word appears in it.
    corpus = [dictionary.doc2bow(token) for token in tokens]
    return dictionary, corpus


def create_lda_model(corpus, dictionary, num_topics=10, passes=10):
    """
    Create a LDA model from a corpus and a dictionary using the gensim library
    Args:
        corpus (list): list of bag of words, i.e number of occurence of each word in the plot
        dictionary (gensim.corpora.dictionary.Dictionary): Dictionary
        num_topics (int, optional): number of topics to extract
        passes (int, optional): number of passes over the corpus
    """
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


def get_topics(lda_model, num_topics=15, num_words=13):
    """Get the topics from a LDA model

    Args:
        lda_model (gensim.models.LdaModel): LdaModel used for topic extraction
        num_topics (int, optional): number of topics we want to extract. Defaults to 15.
        num_words (int, optional): number of words to represent a topic. Defaults to 13.

    Returns:
        list: list of topics, one topic is an array of num_words words
    """
    topics = []
    for _, topic in lda_model.show_topics(formatted=True, num_topics=num_topics, num_words=num_words):
        # we remove the percentage of each word in the topic, and only keep the word
        topics.append(' '.join([word.split('*')[1]
                      for word in topic.split(' + ')]).replace("\"", ''))
    return topics


def get_topic_distribution(lda_model, corpus):
    """ Get the topic distribution for each plot
    Args:
        lda_model (gensim.models.LdaModel): LdaModel used for topic extraction
        corpus (list): list of bag of words, i.e number of occurence of each word in the plot
    Returns:
        list: list of topic distribution for each plot. I.e for each plot, we have a list of tuples
         (topic, probability). We order the list by decreasing probability.
    """
    topic_distributions = []
    for plot in corpus:
        distribution = lda_model.get_document_topics(plot)
        # we sort the list by decreasing probability
        distribution.sort(key=lambda x: x[1], reverse=True)
        topic_distributions.append(distribution)
    return topic_distributions


def print_movie_plot_with_topic_words(topic_distributions, topics, df_plots, num_plots):
    """Print the plots associated to the most probable topic
    Args:
        topic_distributions (list): list of topic distribution for each plot. I.e for each plot,
         we have a list of tuples
        topics (list): list of topics, one topic is an array of num_words words
        df_plots (pandas.DataFrame): dataframe containing the plots
        num_plots (int): number of plots to print
    """
    # TODO add random sampling
    for i, distribution in enumerate(topic_distributions[:num_plots]):
        print('plot: {}'.format(df_plots.iloc[i]['overview']))
        # get the value of the topic with the highest probability
        best_topic = max(distribution, key=lambda x: x[1])[0]
        print('associated best topic: {}'.format(topics[best_topic]))
        print()
