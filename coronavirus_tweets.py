# Part 3: Mining text data.
import pandas as pd
import requests
from nltk.stem import PorterStemmer
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
# Return a pandas dataframe containing the data set.
# Specify a 'latin-1' encoding when reading the data.
# data_file will be populated with the string 'wholesale_customers.csv'.
def read_csv_3(data_file):
    df = pd.read_csv("coronavirus_tweets.csv", encoding='latin-1')
    return df

# Return a list with the possible sentiments that a tweet might have.
def get_sentiments(df):
    return df['Sentiment'].unique()
    

# Return a string containing the second most popular sentiment among the tweets.
def second_most_popular_sentiment(df):
    return df['Sentiment'].value_counts().index[1]
    

# Return the date (string as it appears in the data) with the greatest number of extremely positive tweets.
def date_most_popular_tweets(df):
    return df[df['Sentiment'] == 'Extremely Positive']['TweetAt'].value_counts().index[0]
    

# Modify the dataframe df by converting all tweets to lower case. 
def lower_case(df):
    df['OriginalTweet'] = df["OriginalTweet"].str.lower()
    return df
    

# Modify the dataframe df by replacing each characters which is not alphabetic or whitespace with a whitespace.
def remove_non_alphabetic_chars(df):
    df['OriginalTweet'] = df['OriginalTweet'].replace('[^a-zA-Z\s]+', ' ', regex=True)
    return df

# Modify the dataframe df with tweets after removing characters which are not alphabetic or whitespaces.
def remove_multiple_consecutive_whitespaces(df):
    df['OriginalTweet'] = df['OriginalTweet'].replace('[\s+]', ' ')
    return df


# Given a dataframe where each tweet is one string with words separated by single whitespaces,
# tokenize every tweet by converting it into a list of words (strings).
def tokenize(df):
    df['OriginalTweet'] = df['OriginalTweet'].str.split()
    return df

# Given dataframe tdf with the tweets tokenized, return the number of words in all tweets including repetitions.
def count_words_with_repetitions(tdf):
    return sum(tdf['OriginalTweet'].apply(lambda x: len(x)))

# Given dataframe tdf with the tweets tokenized, return the number of distinct words in all tweets.
def count_words_without_repetitions(tdf):
    unique_words_count = set(word for tweet in tdf['OriginalTweet'] for word in tweet)
    return len(unique_words_count)

# Given dataframe tdf with the tweets tokenized, return a list with the k distinct words that are most frequent in the tweets.
def frequent_words(tdf,k):
    words = [word for tweet in tdf['OriginalTweet'] for word in tweet]
    word_counts = {}
    for word in words:
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1
    sorted_words = sorted(word_counts, key=word_counts.get, reverse=True)
    most_common_words = list(sorted_words)[:k]
    return most_common_words


# Given dataframe tdf with the tweets tokenized, remove stop words and words with <=2 characters from each tweet.
# The function should download the list of stop words via:
# https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt
def remove_stop_words(tdf):
    stopwords = requests.get(
        "https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt").content.decode(
        'utf-8').split("\n")
    df['OriginalTweet'] = df['OriginalTweet'].apply(lambda tweet: [word for word in tweet if word not in stopwords and len(word)>2])
    return df

# Given dataframe tdf with the tweets tokenized, reduce each word in every tweet to its stem.
def stemming(tdf):
    ps = PorterStemmer()
    df['OriginalTweet'] = df['OriginalTweet'].apply(lambda x: [ps.stem(word) for word in x])
    return df

# Given a pandas dataframe df with the original coronavirus_tweets.csv data set,
# build a Multinomial Naive Bayes classifier. 
# Return predicted sentiments (e.g. 'Neutral', 'Positive') for the training set
# as a 1d array (numpy.ndarray). 
def mnb_predict(df):
    corpus = df['OriginalTweet']
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(3, 3))
    X = vectorizer.fit_transform(corpus)
    y = df['Sentiment'].values
    clf = MultinomialNB(alpha=0.001)
    clf.fit(X,y)
    predict = clf.predict(X)
    return predict

# Given a 1d array (numpy.ndarray) y_pred with predicted labels (e.g. 'Neutral', 'Positive') 
# by a classifier and another 1d array y_true with the true labels, 
# return the classification accuracy rounded in the 3rd decimal digit.
def mnb_accuracy(y_pred,y_true):
    return round(accuracy_score(y_true,y_pred),3)
    
