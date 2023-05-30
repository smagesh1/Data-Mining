import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
# Part 2: Cluster Analysis

# Return a pandas dataframe containing the data set that needs to be extracted from the data_file.
# data_file will be populated with the string 'wholesale_customers.csv'.
def read_csv_2(data_file):
    df = pd.read_csv(data_file)
    df = df.drop('Channel',axis=1)
    df = df.drop('Region', axis=1)
    return df



# Return a pandas dataframe with summary statistics of the data.
# Namely, 'mean', 'std' (standard deviation), 'min', and 'max' for each attribute.
# These strings index the new dataframe columns. 
# Each row should correspond to an attribute in the original data and be indexed with the attribute name.
def summary_statistics(df):
    df = df.describe().transpose()
    return df.round()
    


# Given a dataframe df with numeric values, return a dataframe (new copy)
# where each attribute value is subtracted by the mean and then divided by the
# standard deviation for that attribute.
def standardize(df):
    df_copy = df.copy()
    df_copy = (df - df_copy.mean())/df_copy.std()
    return df_copy
    


# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using kmeans.
# y should contain values in the set {0,1,...,k-1}.
def kmeans(df, k):
    df_copy = standardize(df)
    y = pd.Series(KMeans(n_clusters=k).fit(df_copy).labels_)
    return y
    


# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using kmeans++.
# y should contain values from the set {0,1,...,k-1}.
def kmeans_plus(df, k):
    df_copy = standardize(df)
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10)
    kmeans.fit(df_copy)
    y2 = pd.Series(kmeans.predict(df_copy))
    return y2
    


# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using agglomerative hierarchical clustering.
# y should contain values from the set {0,1,...,k-1}.
def agglomerative(df, k):
    df_copy = standardize(df)
    y3 = pd.Series(AgglomerativeClustering(n_clusters=k).fit(df_copy).labels_)
    return y3
    


# Given a data set X and an assignment to clusters y
# return the Solhouette score of the clustering.
def clustering_score(X,y):
    return silhouette_score(X, y)




# Perform the cluster evaluation described in the coursework description.
# Given the dataframe df with the data to be clustered,
# return a pandas dataframe with an entry for each clustering algorithm execution.
# Each entry should contain the: 
# 'Algorithm' name: either 'Kmeans' or 'Agglomerative', 
# 'data' type: either 'Original' or 'Standardized',
# 'k': the number of clusters produced,
# 'Silhouette Score': for evaluating the resulting set of clusters.
def cluster_evaluation(df):
    
    k_values = [3, 5, 10]
    standardzied_data = standardize(df)
    originaldata = df
    data_types = [('Original',originaldata),('Standardized',standardzied_data)]
    results = []
    for i in range(10):
        for data in data_types:
            for k in k_values:
                score = kmeans(data[1], k)
                Silhouette_Score = clustering_score(data[1],score)

                result = {'Algorithm':'kmeans', 'K':k, 'Datatype':data[0], 'Score':Silhouette_Score}
                results.append(result)
    for data in data_types:
        for k in k_values: 
            score = agglomerative(data[1], k)
            Silhouette_Score = clustering_score(data[1],score)

            result = {'Algorithm':'agglormative', 'K':k, 'Datatype':data[0], 'Score':Silhouette_Score}
            results.append(result)
    return pd.DataFrame(results)
    
    

# Given the performance evaluation dataframe produced by the cluster_evaluation function,
#return the best computed Silhouette score.
def best_clustering_score(df): 
    return df['Score'].max()


# Run some clustering algorithm of your choice with k=3 and generate a scatter plot for each pair of attributes.
# Data points in different clusters should appear with different colors.
def scatter_plots(df):
    plt.figure(figsize=(30, 20))
    plt.subplots_adjust(hspace=0.25)
    plt.suptitle("Scatterplots of each pair of attributes", fontsize=28, y=0.95)

    y = kmeans(df,3)
    count=0
    for i in range(0,len(df.columns)):
        for j in range(i+1,len(df.columns)):
            plot = plt.subplot(5, 3, count + 1)
            plt.scatter(df.iloc[:,i], df.iloc[:,j], c=y)
            plot.set_title(f"{df.columns[i]} vs {df.columns[j]}")
            plot.set_xlabel(f"{df.columns[i]}")
            plot.set_ylabel(f"{df.columns[j]}")

            count+=1
    plt.savefig('Scatterplot.pdf') 
    return plt
    



