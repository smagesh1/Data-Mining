# Data-Mining

This repository contains my solutions for the Data Mining coursework at King's College London.

## Task Description

The aim of this coursework assignment was to demonstrate understanding of and obtain experience with classification and cluster analysis, which are among the most important data mining tasks. 

## Files

The repository includes the following files:

- `adult.py`: Python file for the Decision Trees with Categorical Attributes part.
- `wholesale_customers.py`: Python file for the Cluster Analysis part.
- `coronavirus_tweets.py`: Python file for the Mining Text Data part.

## Task Summaries
- Part 1: Decision Trees with Categorical Attributes
  - The aim is to build a decision tree model using the adult data set from the UCI Machine Learning Repository. Some of the goals include:
  - Computing various statistics.
  - Preprocessing the data, for example converting attributes to numeric using one-hot encoding.
  - Building a decision tree model and classifying instances into different income categories.
  - Evaluating the performance of the decision tree model by computing the training error rate.
- Part 2: Cluster Analysis
  - This part involves performing cluster analysis on the wholesale customers data set. Some of the goals include:
  - Computing statistics for each attribute.
  - Dividing the data points into clusters using k-means and agglomerative hierarchical clustering algorithms.
  - Identifying the best set of clusters using the Silhouette score.
- Part 3: Mining Text Data
  - The goal of this part is to analyze and classify sentiment in tweets related to Covid-19. Some of the goals include:
  - Extracting information about the data such as computing the possible sentiments in the tweets and the second most popular sentiment.
  - Creating a sparse representation of the term-document matrix using CountVectorizer.
  - Building a Multinomial Naive Bayes classifier using the provided data set.
  - Determining the training accuracy of the classifier.
  - Tuning the parameters of CountVectorizer to optimize the classification accuracy.

These are some of the aims of the task, all the tasks can be viewed in each python file.
