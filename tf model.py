from sklearn.feature_extraction.text import TfidfVectorizer
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

import os
import pickle
import cPickle
import codecs

from gensim.models.word2vec import LineSentence
path = "/media/tugrulz/Yeni Birim/mldata/subtitles/"
files = os.listdir(path)

sentences = []

vectorizer = TfidfVectorizer(min_df=50, decode_error = "replace", strip_accents = "unicode", stop_words = "english", max_df=0.95, max_features=100, ngram_range=(1, 4),
                                 sublinear_tf=True)
iter = 0

for file in files:
    f = open(path+file, 'rb')
    for line in f.readlines():
        sentences.append(line[1:-2])
    f.close()
    iter +=1
    print(iter)

model = vectorizer.fit(sentences)

file = open('/media/tugrulz/Yeni Birim/mldata/tfidf.pkl', 'wb')
cPickle.dump(model, file)



#
# type(sentences)
# print(sentences[0])
#
#
#
# import pickle
# f = open('/media/tugrulz/Yeni Birim/mldata/tfidf.pkl', 'wb')
# pickle.dump(model, f)
# f.close()
#
#
