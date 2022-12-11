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
#     display_name: Python 3.9.13 ('ada_project')
#     language: python
#     name: python3
# ---

# %%
from src.helper import prepare_df

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
# ![movies](data/movies_ss.png)

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
