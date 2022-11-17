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

TODO

## Methods

### General flow

In order to have a machine learning model to predict which streaming platform a movie is on, we first need to see if we have the data on movies that can discriminate them based on platforms.

Question (1) which aims at defining whether some movie features vary across streaming platforms will provide an answer to this question.  

One of the solutions to predict on which streaming service a movie is found would be to compare this movie with the movies present on each streaming platform (**KNN**). So we need to define a distance function between two movies (more on this in methods for **(3)**).  
We will also consider a linear regression model. One important things to define for this model would be the loss function and dimension used (more on this in methods for **(4)**).
We will try each of these possibilities to see which one produces the best results. In both cases, we can divide our dataset inside a training and a testing part (using cross validation).  
Once the machine learning model is trained, we will then analyze its results on a validation set (the risk being to have overfit our model on our training/testing set, and to have a not very efficient model in reality)


### Datasets

LUDO TIME TO SHINE

### Research Questions Methodology

### **(1)** Features Selection

To answer question 1, we will analyze for each category of our dataset (that we find relevant to analyze), if the distribution of the films according to this category is uniform according to the streaming platforms or not. We will make sure that these distributions are compared in a rigorous way by using statistics on observational studies and hypothesis testing.

### **(2)** NLP over overview

For the extraction of topics we will use LDA. This will involve preparing the movie descriptions in a way that can be interpreted by this model. We will also see if word embedding can be efficient to compare movies based on their description.

### **(3)** Distance Function

Let's now see how we can build a distance function to compare two movies.  
In the choice of this function, we will take into account that this function must minimize the distance between movies belonging to the same streaming service, and ideally maximize the distance between the streaming services. The distance function can also use the distribution of a film according to the chosen topics or word embedding of the movie's overview, obtained from question 2.

### **(4)** Regression Model

The dimensions chosen for this regression model would be the parameters found using question 1.  
We can penalize a movie prediction as soon as the predicted value isn't the correct streaming service. But we can also make something different where the loss is lower if the movie were predicted on a streaming platform that we consider "close" to the expected one.


## Proposed timeline

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
