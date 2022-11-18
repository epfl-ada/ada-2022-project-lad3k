# Streaming Services DNA

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

Streaming services’ market is considerably growing each year. They are revolutionizing the movie industry. For a long period of time, Netflix was dominating the market in all aspects. Now the streaming war is raging on. The King is being overtaken by growing competitors like Disney plus, Amazon Prime and HBO. To come on top of this furious battle, each streaming service is now launching its own content in addition to acquiring old content. For consumers, choosing which streaming platform to subscribe to is becoming harder than ever. Are Streaming services' content becoming more similar? Or is it the opposite? Is each streaming service more specialized on some type of content?

## Research Questions

**Main question**: *Given a movie, is it possible to predict on which streaming platform it will be available?* **(0)**

- Are there features on films(genres, director, film duration,...) that allow to visualize a distinction of the distribution of these films according to the platform of streaming **(1)**
- Can topic analysis be used to infer feelings/adjectives to movies based on their description only? **(2)**
  - If so, does this topic analysis applied to a movie allow to deduce additional information than the one provided by the movie category? **(2.1)**
- Can we define a distance function to compare 2 films? Can this function be still used to compare two films, even if we are missing some features used by the fonction ? **(3)**
- How to define a correct loss function for our classification model, which given a movie, aims to predict the streaming service that will provide it? **(4)**

## Additional Datasets

- [IMDb](https://www.imdb.com/interfaces/): Dataset provided by IMDb, containing information about movies, actors, directors, etc. and their ratings.

- A self-made dataset created with [TMDB](https://www.themoviedb.org/). This new dataset contains most
of the movies that we had in our IMBb dataset, since we created this by querying the TMDB API with
the IMDb ID of each movie. This dataset contains information about the streaming platforms that
provide each movie, as well as the movie's description, budget, revenue, etc.

## Methods

### General flow

We first need to make sure that we have relevant information describing different movies on different platforms and we will have to select the most relevant ones **(1)**. We will use basic statistics to describe different movies on different platforms. We will use NLP to extract useful information from the overview and add them to our relevant features **(2)**.

To predict on which streaming service a movie is on, would be to compare this movie's distance with the movies present on each streaming platform. In order to so, we need to define a distance function between movies **(3)**. We will also have to pre-process these features (depending on their distribution) after selecting the most relevant ones. We may consider other classification models like random forest or Deep learning.

We will split our dataset in three parts: training, testing and validation. We will use the training and testing datasets in order to train our model. Finally we will evalutate its performances on the validation set. We may also use cross-validation if we don't have enough data to train our model. We will repeat this procedure for all candidate models and pick the one with the lowest error. It's essential to define a model evaluation metric to compare our different models (which will be answered by solving **(4)**).



### Datasets

The first self-added dataset was the one that we downloaded from the [IMDb](https://www.imdb.com/interfaces/) website. We downloaded it through a python [script](src/dataset_creation.py) that we wrote.

The second one, is actually a self-made dataset created with [TMDB](https://www.themoviedb.org/). At the beginning, we were thinking of using one available on Kaggle, but we found that it was clearly missing some movies that we had in our IMDb dataset. So we decided to create our own dataset by querying the TMDB API with the IMDb ID of each movie.

Since the greedy implementation of our download script was taking more than 1800 hours to request the API for each movie, we had to find a way to speed up the process. We decided to combine some multi-threading and multiprocessing to make it faster. We had some others issues to save the data since we discovered that in some case `pd.read_csv()` seems to not be bijective with `pd.to_csv()`. But finally, we managed to create our dataset in a reasonable amount of time. (The generated dataset is available [here](data/moviedb_data.tsv.gz) and contains around 290k lines).
### Research Questions Methodology

### **(1)** Platforms description (Features Selection and analysis)

To answer question 1, we will analyze for each category of our dataset, if the distribution of the films according to this category is uniform according to the streaming platforms or not. We will make sure that these distributions are compared in a rigorous way by using statistics on observational studies and hypothesis testing. We will also measure correlations between different features.

### **(2)** NLP over overview

For the extraction of topics we will use LDA. This will involve preparing the movie descriptions in a way that can be interpreted by this model. We will also see if word embedding can be efficient to compare movies based on their description.

### **(3)** Classification Model

In order to define a classification model like KNN, we need to define a distance function to compare movies. In the choice of this function, we will take into account that this function must minimize the distance between movies belonging to the same streaming service, and ideally maximize the distance between the streaming services.

### **(4)** Evaluation metric

Next we need to define a loss function for the training. The score is determined by the distance between the prediction and expected value. This distance will be defined depending on our model and if the predictions are discrete or continuous.

## Proposed timeline

- Platforms description: 18/11 -> 25/11
- NLP over overview: 18/11 -> 25/11
- Classification models: 25/11 -> 12/12
- Evaluation metrics: 25/11 -> 12/12
- Frontend Interface: 12/12 -> 24/12


## Organization within the team

### P2
- Ludovic: Fetching data and using it to construct our two datasets
- Asfoury: Data preprocessing and exploration
- Karim: Data preprocessing and exploration
- Douglas: Exploration of some NLP techniques on the movies' plots

### P3
- Ludovic: Selection of features for models and frontend
- Asfoury: NLP and frontend
- Karim: Platforms description and classification model
- Douglas: NLP and classification model

## Questions: (optional)

In our case, do you have any suggestion on how to define our distance metrics? how to choose our model? our evaluation metric?
