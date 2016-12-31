import re
import numpy as np
from nltk.corpus import stopwords # Import the stop word list
from nltk import PorterStemmer
import os
import sklearn
from sklearn import cross_validation
from termcolor import colored
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.svm import SVC
from sklearn import svm
import collections
import numpy as np
import pandas as pd
import re

import seaborn as sns; sns.set()

def find_titles(tag):
    filepath1 = './tag_relevance.dat'
    filepath2 = './tags.dat'
    filepath3 = './movies.dat'
    relevance_table = pd.read_csv(filepath1, sep='\t', header=None,
                        names=['MovieID', 'TagID', 'TagRelevance'])
    tag_table =pd.read_csv(filepath2, sep='\t', header=None,
                        names=['TagID', 'Tag', 'NumTaggings'])
    #movies_table = movies(filepath3)
    movies = pd.read_csv(filepath3, sep='\t', header=None,
                            names=['MovieID', 'Title', 'NumRatings'])

    #print tag_table.columns.values
    #print tag_table
    row =  tag_table.loc[tag_table['Tag'] == tag]
    tagID =  row.iloc[0]['TagID']
    numTag = row.iloc[0]['NumTaggings']
    #print relevance_table['TagID'] == tagID
    rows = relevance_table.loc[(relevance_table['TagID'] == tagID)] #& (relevance_table['TagRelevance'].values > 0.01)]
    #print rows.shape
    #rows = rows.sort_values('TagRelevance', ascending = False)
    #print rows['TagRelevance']
    ids = rows.ix[:,'MovieID':'MovieID']
    #print ids.iloc[1]['MovieID']
    titles = list()
    amount = []
    #print relevance_table.shape

    # for testing
    numTag = 9374
    for i in range(numTag):
        #titles.append(movies.loc[movies['MovieID'] == ids.iloc[i]['MovieID']].iloc[0]['Title'])
        amount.append(rows.iloc[i]['TagRelevance'])
    #title = set(titles)

    return amount


tags = []
f = open('master-tags.txt', 'rb')
for line in f.readlines():
    if(len(line) > 4):
        tag = line.split('\t')[1]
        print(tag)
        tags.append(tag)

data = []
for i in range(0,len(tags)):
    amount = find_titles(tags[i])
    data.append(amount)

y = [i for j in range(0,len(data))]

predicted = find_titles("bleak")

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(data, y)

import cPickle
f = open('data.pkl', 'wb')
cPickle.dump(data, f)
f.close()

print("Guessed " + tags[neigh.predict(predicted)])

#869	sacrifice	29


