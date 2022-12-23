# Streaming Services DNA


## Link to our data story

[Data Story](http://data-story.best-ada-project.ch/)

## Installation
```bash
conda create -y -n ada_project python=3.9 pip
conda activate ada_project
pip install -r requirements.txt
pre-commit install
```

To not track the changes in config.py, you can use the following command:
```bash
 git update-index --assume-unchanged src/config.py
```


## abstract

Streaming services’ market is considerably growing each year. They are revolutionizing the movie industry. For a long period of time, Netflix was dominating the market in all aspects. Now the streaming war is raging on. The King is being overtaken by growing competitors like Disney plus, Amazon Prime and HBO. To come on top of this furious battle, each streaming service is now launching its own content in addition to acquiring old content. For consumers, choosing which streaming platform to subscribe to is becoming harder than ever. Our goal is to provide insights on which platform offers the best value for movie lovers and help users make informed decisions when choosing a streaming service. We will focus on analyzing movies available on Netflix and Amazon Prime to determine which platform offers a better selection. We will be using data on movie ratings, genre, and other relevant features to compare the two platforms.

## Research Questions

**Main question**: *Which streaming service platform offers better movies? Prime or Netflix?* **(0)**

- Are there features on films(genres, production countrie, average rating, film duration,...) that allow to visualize a distinction of the distribution of these films according to the platform of streaming ? **(1)**
- Can topic analysis be used to infer feelings/adjectives to movies based on their description only? **(2)**
  - If so, does this topic analysis applied to a movie allow to deduce additional information than the one provided by the movie category? **(2.1)**
- Is there a significant difference in the quality of movies between the two platforms? Is there a causal effect between being on a platform and having higher average rating?


### Datasets

- [IMDb](https://www.imdb.com/interfaces/): Dataset provided by IMDb, containing information about movies, actors, directors, etc. and their ratings.

- A self-made dataset created with [TMDB](https://www.themoviedb.org/). This new dataset contains most
of the movies that we had in our IMBb dataset, since we created this by querying the TMDB API with
the IMDb ID of each movie. This dataset contains information about the streaming platforms that
provide each movie, as well as the movie's description, budget, revenue, etc.

The first self-added dataset was the one that we downloaded from the [IMDb](https://www.imdb.com/interfaces/) website. We downloaded it through a python [script](src/dataset_creation.py) that we wrote.

The second one, is actually a self-made dataset created with [TMDB](https://www.themoviedb.org/). At the beginning, we were thinking of using one available on Kaggle, but we found that it was clearly missing some movies that we had in our IMDb dataset. So we decided to create our own dataset by querying the TMDB API with the IMDb ID of each movie.

Since the greedy implementation of our download script was taking more than 1800 hours to request the API for each movie, we had to find a way to speed up the process. We decided to combine some multi-threading and multiprocessing to make it faster. We had some others issues to save the data since we discovered that in some case `pd.read_csv()` seems to not be bijective with `pd.to_csv()`. But finally, we managed to create our dataset in a reasonable amount of time. (The generated dataset is available [here](data/moviedb_data.tsv.gz) and contains around 290k lines).

## Methods

- Data collection: We will scrape movie data from Netflix and Amazon Prime using web scraping techniques.
- Data cleaning and processing: We will clean and process the data to prepare it for analysis. This will involve removing any irrelevant or incomplete entries, and ensuring that all data is in a consistent format.
- Data visualization: We will use data visualization techniques to visualize and compare the movies available on the two platforms.
- Data analysis: We will use statistical methods to analyze the data and answer our research questions. This includes hypothesis testing, observational studies. Also new features will be created using NLP to extract useful information from the overview.

## Proposed timeline:

Week 1: Scrape movie data from Netflix and Amazon Prime  
Week 2: Clean and process the data  
Week 3: Visualize the data and perform initial analysis  
Week 4: Finalize analysis and write report  

## Organization within the team

### P2

- Asfoury: Data preprocessing and exploration
- Karim: Data preprocessing and exploration
- Ludovic: Fetching data and using it to construct our two datasets
- Douglas: Exploration of some NLP techniques on the movies' plots

### P3

- Asfoury: Data cleaning and website creation  
- Karim: Data cleaning, visualization and initial analysis  
- Ludovic: Data collection, Observational study  
- Douglas: NLP, Observational study  
- Everyone: Website story and explanation writing
