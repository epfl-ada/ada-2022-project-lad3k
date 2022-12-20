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
from src import helper
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import statsmodels.formula.api as smf
from sklearn.metrics import classification_report
import networkx as nx
from tqdm import tqdm
import math


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
                 'release_year', 'runtimeMinutes', 'streaming_service']], hue='streaming_service')
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

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for i, col in enumerate(['numVotes', 'release_year', 'runtimeMinutes']):
        if col in ['release_year', 'runtimeMinutes']:
            axs[i].hist(df_netflix[col], alpha=0.5,
                        label='Netflix', density=True, bins=20)
            axs[i].hist(df_prime[col], alpha=0.5,
                        label='Prime', density=True, bins=20)
            axs[i].set_ylabel('density')
        elif col in ['numVotes']:

            max_x_value = max(df_netflix[col].max(), df_prime[col].max())
            bins_logspace = np.logspace(0, np.log10(max_x_value), 40)

            axs[i].hist(df_netflix[col], alpha=0.5,
                        label='Netflix', bins=bins_logspace)
            axs[i].hist(df_prime[col], alpha=0.5,
                        label='Prime', bins=bins_logspace)
            axs[i].set_xscale('log')
            axs[i].set_ylabel('number of movies')
        else:
            # should never happen
            raise ValueError('column not found')
        axs[i].legend()
        axs[i].set_title(col)
    # axs[-1, -1].axis('off')  # hide last subplot as nothing in it
    plt.show()


plot_hist_matching(df_netflix, df_prime)


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
plot_rating_distribution(balanced_df)

# %%
df_netflix.columns

# %%
plot_hist_matching(df_netflix, df_prime)

# %%
plot_genre_distribution(df_netflix, df_prime)

# %%