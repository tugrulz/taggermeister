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
from patsy import dmatrices
import os.path

import seaborn as sns; sns.set()



path = '/media/tugrulz/Yeni Birim/mldata/subtitles/'
def find_titles(tag, addition = 0, all = False):
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
    numTag = row.iloc[0]['NumTaggings'] + addition
    #print relevance_table['TagID'] == tagID
    rows = relevance_table.loc[(relevance_table['TagID'] == tagID)] #& (relevance_table['TagRelevance'].values > 0.01)]
    #print rows.shapes
    rows = rows.sort('TagRelevance', ascending = False)
    #print rows['TagRelevance']
    ids = rows.ix[:,'MovieID':'MovieID']
    #print ids.iloc[1]['MovieID']
    titles = list()
    #print relevance_table.shape
    overhead = 0
    i = 0
    while (i < numTag):
        name = movies.loc[movies['MovieID'] == ids.iloc[i+overhead]['MovieID']].iloc[0]['Title']
        if(name[0:5] == "8 1/2"):
            name = "8 1slash2 (8slash) (1963)"
        else:
            name = name.replace('/', "slash")
        if(os.path.isfile(path + name + ".txt")):
            titles.append((name, rows.iloc[i+overhead]['TagRelevance']))
            print(titles[-1])
            i += 1
        else:
            #print("No subtitle for " + name + " of iteration " + str(i+overhead))
            overhead += 1
    #title = set(titles)

    return titles, titles[-1][1]

def find_worst_titles(tag, addition = 0):
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
    numTag = row.iloc[0]['NumTaggings'] + addition
    #print relevance_table['TagID'] == tagID
    rows = relevance_table.loc[(relevance_table['TagID'] == tagID)] #& (relevance_table['TagRelevance'].values > 0.01)]
    #print rows.shapes
    rows = rows.sort('TagRelevance', ascending = True)
    #print rows['TagRelevance']
    ids = rows.ix[:,'MovieID':'MovieID']
    #print ids.iloc[1]['MovieID']
    titles = list()
    #print relevance_table.shape
    overhead = 0
    i = 0
    while (i < numTag):
        name = movies.loc[movies['MovieID'] == ids.iloc[i+overhead]['MovieID']].iloc[0]['Title']
        if(name[0:5] == "8 1/2"):
            name = "8 1slash2 (8slash) (1963)"
        else:
            name = name.replace('/', "slash")
        if(os.path.isfile(path + name + ".txt")):
            titles.append((name, rows.iloc[i+overhead]['TagRelevance']))
            print(titles[-1])
            i += 1
        else:
            #print("No subtitle for " + name + " of iteration " + str(i+overhead))
            overhead += 1
    #title = set(titles)
    return titles, titles[-1][1]

def findAllFiles(directory):
    files = []
    for f in os.listdir(directory):
        files.append(directory + "/" + f)
    return files

def get_labels(files):
    labels = []
    for file in files:
       labels.append(file.split("/")[2].split("-")[0])
    return labels

def get_given_labels(files, label):
    #labels = []
    labels = np.zeros(shape=(len(files),1), dtype=int)
    titles = find_titles(label)
    for file in files:
        if file in titles:
            labels[len(labels)-1] = 1
        else:
            labels[len(labels)-1] = 0
    return labels

def getTextOfFile(file):
    with open(file, 'r') as myfile:
     return myfile.read().replace('\n', '')

def create_bag_of_words(files, save = False):
    print (" bag of words is being created...")
    subtitles = []
    for file in files:
        subtitles.append(getTextOfFile(file))
    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.
    vectorizer = CountVectorizer(analyzer = "word",
                                 tokenizer = None,
                                 preprocessor = None,
                                 stop_words = None,
                                 max_features = 1000)

    #vectorizer = TfidfVectorizer(input  = subtitles,
    #                             analyzer = "word",
    #                             tokenizer = None,
    #                             preprocessor = None,
    #                             stop_words = None,
    #                             max_features = 500)
    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of
    # strings.
    features = vectorizer.fit_transform(subtitles)

    # Numpy arrays are easy to work with, so convert the result to an
    # array
    if (save):
        np.savetxt("bag_of_words.csv", features.toarray(), fmt='%i', delimiter = ',')
    return np.asmatrix(features.toarray(), dtype=int)


def subtitle_to_words( subtitle ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    # review_text = BeautifulSoup(subtitle).get_text()
    # subtitle = review_text
    #
    # 2. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", subtitle)
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))
    #
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]

    # 5.5 Stem
    stemmer = PorterStemmer()
    meaningful_words = [stemmer.stem(word) for word in meaningful_words]

    #
    # 6. Join the words back into one string separated by space,
    # and return the result.
    return( " ".join( meaningful_words ))

def process_words(files, new_folder):
    print ("words are processing...")
    for file in files:
        text = getTextOfFile(file)
        processed_text = subtitle_to_words(text)

        newFile = file.replace("data", new_folder)
        with open(newFile, 'a') as f:
          f.write(processed_text)

def test_single(clf, trainingX, trainingY, X):
    clf.fit(trainingX, trainingY)
    return clf.predict(X)

def plot_svc_decision_function(clf, ax = None):
    X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(X, Y, test_size=0.3)
    clf.fit(X_train, Y_train)
    if ax is None:
        plt.gca()

    Y, X = np.meshgrid(y, x)
    P = np.zeros_like(X_test)
    for i, xi in enumerate(X_test):
        for j, yj in enumerate(y):
            P[i, j] = clf.decision_function([xi, yj])
    # plot the margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])


def test_classifier(X, y, clf, test_size=0.4, y_names=None, confusion=False):
	#train-test split
	print 'test size is: %2.0f%%' % (test_size*100)
	X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(X, y, test_size=test_size)

	clf.fit(X_train, y_train)
	y_predicted = clf.predict(X_test)

	if not confusion:
		print colored('Classification report:', 'magenta', attrs=['bold'])
		print sklearn.metrics.classification_report(y_test, y_predicted, target_names=y_names)
	else:
		print colored('Confusion Matrix:', 'magenta', attrs=['bold'])
		print sklearn.metrics.confusion_matrix(y_test, y_predicted)


