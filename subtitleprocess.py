import os

path = "/media/tugrulz/Yeni Birim/mldata/subtitles"
newpath = "/media/tugrulz/Yeni Birim/mldata/processed"

import re
import numpy as np
from nltk.corpus import stopwords  # Import the stop word list
from nltk import PorterStemmer
import os
from sklearn.feature_extraction.text import CountVectorizer


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


def getTextOfFile(file):
    with open(file, 'r') as myfile:
        return myfile.read().replace('\n', '')


def create_bag_of_words(files, save=False):
    print (" bag of words is creating...")
    subtitles = []
    for file in files:
        subtitles.append(getTextOfFile(file))
    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.
    vectorizer = CountVectorizer(analyzer="word",
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 max_features=5000)

    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of
    # strings.
    features = vectorizer.fit_transform(subtitles)

    # Numpy arrays are easy to work with, so convert the result to an
    # array
    if (save):
        np.savetxt("bag_of_words.txt", features.toarray(), fmt='%i')
    return features.toarray()


def subtitle_to_words(subtitle):
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
    return (" ".join(meaningful_words))


def process_words(files, newpath):
    print ("words are processing...")
    for file in files:
        text = getTextOfFile(file)
        processed_text = subtitle_to_words(text)
        print(file)
        with open(newpath+"/"+file.split('/')[-1], 'a') as f:
            f.write(processed_text)


files = findAllFiles(path)

process_words(files, newpath)




#files = findAllFiles("./data_processed")
#labels = get_labels(files)
#bag_of_words = create_bag_of_words(files, save=False)  # bag of words could be saved to a txt too



