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
#     display_name: ada_hw2
#     language: python
#     name: python3
# ---
# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import bootstrap
from scipy.stats import ttest_ind

from src.helper import prepare_df
# %%

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
both_genre.plot(kind='bar', figsize=(15, 5))


plt.title('Movies genre on Netflix and Prime')
plt.xlabel('Genre')
plt.ylabel('Normalized Frequency')
plt.show()


# %%

both_genre = both_genre[:10]
both_genre = both_genre.reset_index()
both_genre = both_genre.sample(frac=1)


fig = go.Figure()

fig.add_trace(go.Scatterpolar(
    r=both_genre['nb_movies_netflix'],
    theta=both_genre.index,
    fill='toself',
    name='Netflix',

))
fig.add_trace(go.Scatterpolar(
    r=both_genre['nb_movies_prime'],
    theta=both_genre.index,
    fill='toself',
    name='Prime',
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

    # plot the movie rating distribution on Netflix and Prime
    # rating is in column "averageRating"
    # a col "streaming_service" tells us if the movie is on Netflix, Prime or both

    if column == 'averageRating':
        plt.hist(df[df['on_netflix']][column],
                 bins=np.arange(0, 10.1, 0.5),
                 alpha=0.5,
                 density=True,
                 color='C0',
                 label='Netflix')
        plt.axvline(df[df['on_netflix']][column].mean(),
                    color='C0', linestyle='dashed', linewidth=1)
        plt.axvline(df[df['on_netflix']][column].median(),
                    color='C0', linestyle='dotted', linewidth=1)

        plt.hist(df[df['on_prime']][column],
                 bins=np.arange(0, 10.1, 0.5),
                 alpha=0.5,
                 density=True,
                 color='C1',
                 label='Prime')
        plt.axvline(df[df['on_prime']][column].mean(),
                    color='C1', linestyle='dashed', linewidth=1)
        plt.axvline(df[df['on_prime']][column].median(),
                    color='C1', linestyle='dotted', linewidth=1)

        # add legend for hist and mean/median
        legend_elements = [
            plt.Line2D([0], [0], color='C0', lw=6, label='Netflix'),
            plt.Line2D([0], [0], color='C1', lw=6, label='Prime'),
            plt.Line2D([0], [0], color='k', lw=3,
                       linestyle='dashed', label='Mean'),
            plt.Line2D([0], [0], color='k', lw=3,
                       linestyle='dotted', label='Median')
        ]
        plt.legend(handles=legend_elements, loc='upper right')

        plt.title('Movie rating distribution on Netflix and Prime')
        plt.xlabel('Average rating')
        plt.ylabel('Density')
        plt.show()

    if column == 'runtimeMinutes':
        plt.hist(df[df['on_netflix']][column],
                 bins=np.arange(0, 350, 10),
                 alpha=0.5,
                 density=True,
                 color='C0',
                 label='Netflix')
        plt.axvline(df[df['on_netflix']][column].mean(),
                    color='C0', linestyle='dashed', linewidth=1)
        plt.axvline(df[df['on_netflix']][column].median(),
                    color='C0', linestyle='dotted', linewidth=1)

        plt.hist(df[df['on_prime']][column],
                 bins=np.arange(0, 350, 10),
                 alpha=0.5,
                 density=True,
                 color='C1',
                 label='Prime')
        plt.axvline(df[df['on_prime']][column].mean(),
                    color='C1', linestyle='dashed', linewidth=1)
        plt.axvline(df[df['on_prime']][column].median(),
                    color='C1', linestyle='dotted', linewidth=1)

        # add legend for hist and mean/median
        legend_elements = [
            plt.Line2D([0], [0], color='C0', lw=6, label='Netflix'),
            plt.Line2D([0], [0], color='C1', lw=6, label='Prime'),
            plt.Line2D([0], [0], color='k', lw=3,
                       linestyle='dashed', label='Mean',),
            plt.Line2D([0], [0], color='k', lw=3,
                       linestyle='dotted', label='Median')
        ]
        plt.legend(handles=legend_elements, loc='upper right')

        plt.title('Movie run time distribution on Netflix and Prime')
        plt.xlabel('run rime (min)')
        plt.ylabel('Density ')
        plt.show()

    if column == 'release_year':
        plt.hist(df[df['on_netflix']][column],
                 bins=20,
                 alpha=0.5,
                 density=True,
                 color='C0',
                 label='Netflix',
                 log=True)
        plt.axvline(df[df['on_netflix']][column].mean(),
                    color='C0', linestyle='dashed', linewidth=1)
        plt.axvline(df[df['on_netflix']][column].median(),
                    color='C0', linestyle='dotted', linewidth=1)

        plt.hist(df[df['on_prime']][column],
                 bins=20,
                 alpha=0.5,
                 density=True,
                 color='C1',
                 label='Prime',
                 log=True)
        plt.axvline(df[df['on_prime']][column].mean(),
                    color='C1', linestyle='dashed', linewidth=1)
        plt.axvline(df[df['on_prime']][column].median(),
                    color='C1', linestyle='dotted', linewidth=1)

        # add legend for hist and mean/median
        legend_elements = [
            plt.Line2D([0], [0], color='C0', lw=6, label='Netflix'),
            plt.Line2D([0], [0], color='C1', lw=6, label='Prime'),
            plt.Line2D([0], [0], color='k', lw=3,
                       linestyle='dashed', label='Mean',),
            plt.Line2D([0], [0], color='k', lw=3,
                       linestyle='dotted', label='Median')
        ]
        plt.legend(handles=legend_elements, loc='upper right')

        plt.title('Movie release year distribution on Netflix and Prime')
        plt.xlabel('release year')
        plt.ylabel('Density ')
        plt.show()

    if column == 'revenue':
        plt.hist(df[df['on_netflix']][column],
                 bins=20,
                 alpha=0.5,
                 color='C0',
                 density=True,
                 label='Netflix',
                 log=True)
        plt.axvline(df[df['on_netflix']][column].mean(),
                    color='C0', linestyle='dashed', linewidth=1)
        plt.axvline(df[df['on_netflix']][column].median(),
                    color='C0', linestyle='dotted', linewidth=1)

        plt.hist(df[df['on_prime']][column],
                 bins=20,
                 alpha=0.5,
                 color='C1',
                 density=True,
                 label='Prime',
                 log=True)
        plt.axvline(df[df['on_prime']][column].mean(),
                    color='C1', linestyle='dashed', linewidth=1)
        plt.axvline(df[df['on_prime']][column].median(),
                    color='C1', linestyle='dotted', linewidth=1)

        # add legend for hist and mean/median
        legend_elements = [
            plt.Line2D([0], [0], color='C0', lw=6, label='Netflix'),
            plt.Line2D([0], [0], color='C1', lw=6, label='Prime'),
            plt.Line2D([0], [0], color='k', lw=3,
                       linestyle='dashed', label='Mean',),
            plt.Line2D([0], [0], color='k', lw=3,
                       linestyle='dotted', label='Median')
        ]
        plt.legend(handles=legend_elements, loc='upper right')

        plt.title('Movies revenues distribution on Netflix and Prime')
        plt.xlabel('revenue ($)')
        plt.ylabel('Density ')
        plt.show()

    if column == 'numVotes':
        plt.hist(df[df['on_netflix']][column],
                 bins=20,
                 alpha=0.5,
                 color='C0',
                 density=True,
                 label='Netflix',
                 log=True)
        plt.axvline(df[df['on_netflix']][column].mean(),
                    color='C0', linestyle='dashed', linewidth=1)
        plt.axvline(df[df['on_netflix']][column].median(),
                    color='C0', linestyle='dotted', linewidth=1)

        plt.hist(df[df['on_prime']][column],
                 bins=20,
                 alpha=0.5,
                 color='C1',
                 density=True,
                 label='Prime',
                 log=True)
        plt.axvline(df[df['on_prime']][column].mean(),
                    color='C1', linestyle='dashed', linewidth=1)
        plt.axvline(df[df['on_prime']][column].median(),
                    color='C1', linestyle='dotted', linewidth=1)

        # add legend for hist and mean/median
        legend_elements = [
            plt.Line2D([0], [0], color='C0', lw=6, label='Netflix'),
            plt.Line2D([0], [0], color='C1', lw=6, label='Prime'),
            plt.Line2D([0], [0], color='k', lw=3,
                       linestyle='dashed', label='Mean',),
            plt.Line2D([0], [0], color='k', lw=3,
                       linestyle='dotted', label='Median')
        ]
        plt.legend(handles=legend_elements, loc='upper right')

        plt.title('Movies numner of votes on Netflix and Prime')
        plt.xlabel('number of votes')
        plt.ylabel('Density ')
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
# mean of the avergae rating movies on Prime. Movies on Netfliy are generally higher rated than movies on Prime.

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
fig = px.scatter_geo(df_prod_countrie, locations='production_countries', color='streaming_service',
                     hover_name='production_countries', size='size',
                     projection='natural earth')


fig.show()

# fig.write_html("worldmap.html")

# %%
df_prod_countrie

# %%
test = (df_prod_countrie['Netflix'],)
bootstrap(test, np.median, method='percentile').confidence_interval.low


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
    10, 5), capsize=5)


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
    10, 5))

# %%
# df_prod_comp = df_prod_comp[(df_prod_comp['Netflix'] > 0) & (df_prod_comp['Prime'] > 0)]

# df_prod_comp[:20].plot(kind='bar', title='Number of movies per production company in Netflix and Prime',
# figsize=(10, 5))

# %% [markdown]
# > There are 900 production companies in common out of 6286. They have very different prodcution companies.

# %%
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

df_writers[:20].plot(
    kind='bar', title='Number of movies per writer in Netflix and Prime', figsize=(10, 5))

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

df_directors[:20].plot(
    kind='bar', title='Number of movies per director in Netflix and Prime', figsize=(10, 5))
