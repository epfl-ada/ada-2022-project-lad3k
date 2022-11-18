# TO RENAME - Lad3k

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


# Title

## abstract

Pk on veut faire ca


## Research Questions

**Main question**: *Given a movie, is it possible to predict on which streaming platform it will be available?* **(0)**

- Are there features on films(genres, director, film duration,...) that allow to visualize a distinction of the distribution of these films according to the platform of streaming **(1)**
- Can topic analysis be used to infer feelings/adjectives to movies based on their description only? **(2)**
  - If so, does this topic analysis applied to a movie allow to deduce additional information than the one provided by the movie category? **(2.1)**
- Can we define a distance function to compare 2 films? Does this function take into account the categories verifying the previous question, or other categories? Can this function be still used to compare two films, even if we are missing some features used by the fonction ? **(3)**
- How to define a correct loss function for a regression model, which given a movie, aims at predict the streaming service that will provide it. **(4)**

## Additional Datasets

- [IMDb](https://www.imdb.com/interfaces/): Dataset provided by IMDb, containing information about movies, actors, directors, etc. and their ratings.

- A self-made dataset created with [TMDB](https://www.themoviedb.org/). This new dataset contains most
of the movies that we had in our IMBb dataset, since we created this by querying the TMDB API with
the IMDb ID of each movie. This dataset contains information about the streaming platforms that
provide each movie, as well as the movie's description, budget, revenue, etc.

## Methods

### General flow

In order to have a machine learning model to predict which streaming platform a movie is on, we first need to see if we have the data on movies that can discriminate them based on platforms.

Question (1) which aims at defining whether some movie features vary across streaming platforms will provide an answer to this question.  

One of the solutions to predict on which streaming service a movie is found would be to compare this movie with the movies present on each streaming platform (**KNN**). So we need to define a distance function between two movies (more on this in methods for **(3)**).  
We will also consider a **regression model**. One important things to define for this model would be the loss function and dimension used (more on this in methods for **(4)**).
In both cases, we can divide our dataset inside a training and a testing part (using cross validation). We will also leave some movies to create a validation set. We use as validation set as there is a risk of overfitting our model on our training/testing set, and to have a not very efficient model in reality. Thus validation set allows to test our model on data it has never seen before.  
Finally we will see which one of these two models leads the best results on our validation set.


### Datasets

The first self-added dataset was the one that we downloaded from the [IMDb](https://www.imdb.com/interfaces/) website. We downloaded it through a python [script](src/dataset_creation.py) that we wrote.

The second one, is actually a self-made dataset created with [TMDB](https://www.themoviedb.org/). At the beginning, we were thinking of using one available on Kaggle, but we found that it was clearly missing some movies that we had in our IMDb dataset. So we decided to create our own dataset by querying the TMDB API with the IMDb ID of each movie.

Since the greedy implementation of our download script was taking more than 1800 hours to request the API for each movie, we had to find a way to speed up the process. We decided to combine some multi-threading and multiprocessing to make it faster. We had some others issues to save the data since we discovered that in some case `pd.read_csv()` seems to not be bijective with `pd.to_csv()`. But finally, we managed to create our dataset in a reasonable amount of time. (The generated dataset is available [here](data/moviedb_data.tsv.gz) and contains around 290k lines).
### Research Questions Methodology

### **(1)** Features Selection

To answer question 1, we will analyze for each category of our dataset (that we find relevant to analyze), if the distribution of the films according to this category is uniform according to the streaming platforms or not. We will make sure that these distributions are compared in a rigorous way by using statistics on observational studies and hypothesis testing.

### **(2)** NLP over overview

For the extraction of topics we will use LDA. This will involve preparing the movie descriptions in a way that can be interpreted by this model. We will also see if word embedding can be efficient to compare movies based on their description.

### **(3)** KNN

In order to define a KNN, we need to define a distance function comparing two movies.
In the choice of this function, we will take into account that this function must minimize the distance between movies belonging to the same streaming service, and ideally maximize the distance between the streaming services. The distance function can also use the distribution of a film according to the chosen topics or word embedding of the movie's overview, obtained from question 2.

### **(4)** Regression Model

The dimensions chosen for this regression model would be the parameters found using question 1 and also topics distribution from question 2.  
Next we need to define a loss function for the training.
We can penalize a movie prediction as soon as the predicted value isn't the correct streaming service. But we can also make something different where the loss is lower if the movie were predicted on a streaming platform that we consider "close" to the expected one. These are choices that we can try, and see which one leads to the better results on the test set.



## Proposed timeline

TODO

- Features Selection: 18/11 -> 30/11
- NLP over overview: 18/11 -> 30/11
- KNN: 01/12 -> 24/12
- Regression Model: 01/12 -> 24/12



## Organization within the team

### P2
- Ludovic: Fetching data and using it to construct our two datasets
- Asfoury: TBD
- Karim: TBD
- Douglas: Exploration of some NLP techniques on the movies' plots

### P3
- Ludovic: TBD
- Asfoury: TBD
- Karim: TBD
- Douglas: TBD

## Questions: (optional)
