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


import dias
from dias import *

tag = "freedom"
add  = 16
test_start = 40


def build_vectorizer (tag, force = False, addition = add):
    if ((os.path.isfile('/media/tugrulz/Yeni Birim/mldata/tfidf_' + tag + '.pkl') == True) & (force == False)):
        return
    else:
        iter = 0
        vectorizer = TfidfVectorizer(min_df=50, decode_error="ignore", strip_accents="ascii", stop_words="english",
                                     max_df=0.95, max_features=1000,
                                     sublinear_tf=True)
        best, threshold1 = find_titles(tag, addition)
        worst, threshold2 = find_worst_titles(tag, addition)
        for tuple in best:
            file = tuple[0]
            if (file[0:3] == "8 1"):
                file = "8 1slash2 (8slash) (1963)"
            else:
                file = file.replace("/", "slash")
            f = open(path+file+".txt", 'rb')
            for line in f.readlines():
                sentences.append(line[1:-2])
            f.close()
            iter +=1
            print(iter)

        for tuple in worst:
            file = tuple[0]
            if (file[0:3] == "8 1"):
                file = "8 1slash2 (8slash) (1963)"
            else:
                file = file.replace("/", "slash")
            f = open(path+file+".txt", 'rb')
            for line in f.readlines():
                sentences.append(line[1:-2])
            f.close()
            iter +=1
            print(iter)
        model = vectorizer.fit(sentences)
        file = open('/media/tugrulz/Yeni Birim/mldata/tfidf_' + tag + '.pkl', 'wb')
        cPickle.dump(model, file)
        file.close()

build_vectorizer(tag, add)

file = open('/media/tugrulz/Yeni Birim/mldata/tfidf_' + tag + '.pkl', 'rb')
vectorizer = cPickle.load(file)

print(vectorizer.get_feature_names())

best, threshold1 = find_titles(tag, add)
worst, threshold2 = find_worst_titles(tag, add)

best_labels = [1 for b in best]
worst_labels = [0 for w in worst]

data_dict = {}

best_data = []
for tuple in best:
    file = tuple[0]
    if (file[0:3] == "8 1"):
        file = "8 1slash2 (8slash) (1963)"
    else:
        file = file.replace("/", "slash")
    f = open(path + file + ".txt", 'rb')
    index = len(best_data)
    data_dict[file] = (index, "best")
    best_data.append(f.read().replace('\n',''))
    f.close()

worst_data = []
for tuple in worst:
    file = tuple[0]
    if (file[0:3] == "8 1"):
        file = "8 1slash2 (8slash) (1963)"
    else:
        file = file.replace("/", "slash")
    f = open(path + file + ".txt", 'rb')
    index = len(worst_data)
    data_dict[file] = (index, "worst")
    worst_data.append(f.read().replace('\n',''))
    f.close()

from random import shuffle
shuffle(best_data)
shuffle(worst_data)

train_data_features = vectorizer.transform( best_data[0:test_start] + worst_data[0:test_start] )
print(vectorizer.get_params())
print()
print(train_data_features[0])

# import numpy as np
#
# indices = np.argsort(vectorizer.idf_)[::-1]
# features = vectorizer.get_feature_names()
# top_n = 100
# top_features = [features[i] for i in indices[:top_n]]
# str = ""
# iter = 0;
# for f in top_features:
#     str += f + " "
#     iter += 1
#     if (iter % 10 == 0):
#         str += "\n"
#
# print(str)

#####

from collections import defaultdict

features_by_gram = defaultdict(list)
for f, w in zip(vectorizer.get_feature_names(), vectorizer.idf_):
    features_by_gram[len(f.split(' '))].append((f, w))
top_n = 100
for gram, features in features_by_gram.iteritems():
    top_features = sorted(features, key=lambda x: x[1], reverse=True)[:top_n]
    top_features = [f[0] for f in top_features]
    print '{}-gram top:'.format(gram), top_features
    iter = 0
    seter = ""
    for f in top_features:
        seter += f + " , "
        iter += 1
        if (iter % 10 == 0):
            seter += "\n"
print seter


test_data_features = vectorizer.transform( best_data[test_start:] + worst_data[test_start:])

print "Reducing dimension..."

from sklearn.feature_selection.univariate_selection import SelectKBest, chi2, f_classif
fselect = SelectKBest(chi2 , k=200)
train_data_features = fselect.fit_transform(train_data_features, best_labels[0:test_start] + worst_labels[0:test_start])
test_data_features = fselect.transform(test_data_features)

print "Training..."

from sklearn.linear_model import SGDClassifier
#from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

model1 = MLPClassifier()
model1.fit( train_data_features.toarray(), best_labels[0:test_start] + worst_labels[0:test_start] )

model2 = SGDClassifier(loss='modified_huber', n_iter=5, random_state=0, shuffle=True)
model2.fit( train_data_features, best_labels[0:test_start] + worst_labels[0:test_start] )

model3 = SVC(kernel='linear', max_iter = 5000, probability=True)
model3.fit( train_data_features, best_labels[0:test_start] + worst_labels[0:test_start] )

from sklearn import tree
model4 = tree.DecisionTreeClassifier()
model4.fit( train_data_features, best_labels[0:test_start] + worst_labels[0:test_start] )

p1 = model1.predict_proba( test_data_features )#[:,1]
p2 = model2.predict_proba( test_data_features )#[:,1]
p3 = model3.predict_proba( test_data_features )
p4 = model3.predict_proba( test_data_features )

rangers = len(best_data) - test_start
print("Range is " + str(rangers))
labels = [1 for i in range(0, rangers)]
labels2 = [0 for i in range(0, rangers)]

labels = labels + labels2



print("Length of Training data with positive label:")
print(test_start)
print("Length of Training data with negative label:")
print(test_start)
print("Length of Test data with positive label:")
print(len(best_data[test_start:]))
print("Length of Test data with negative label:")
print(len(worst_data[test_start:]))


acc_p1 = 0
acc_p2 = 0
acc_p3 = 0
acc_p4 = 0

acc2_p1 = 0
acc2_p2 = 0
acc2_p3 = 0
acc2_p4 = 0

tp = [[0,0,0,0], [0,0,0,0]] # high-low threshold
tn = [[0,0,0,0], [0,0,0,0]]
fn = [[0,0,0,0], [0,0,0,0]]
fp = [[0,0,0,0], [0,0,0,0]]

for i in range(0,rangers):
    if (p1[i][1] > max(threshold1, 0.7)):
        acc_p1 += 1
        tp[0][0] += 1
    else:
        fp[0][0] += 1
    if (p2[i][1] > max(threshold1, 0.7)):
        acc_p2 += 1
        tp[0][1] += 1
    else:
        fp[0][1] += 1
    if (p3[i][1] > max(threshold1, 0.7)):
        acc_p3 += 1
        tp[0][2] += 1
    else:
        fp[0][2] += 1
    if (p4[i][1] > max(threshold1, 0.7)):
        acc_p4 += 1
        tp[0][3] += 1
    else:
        fp[0][3] += 1


for i in range(rangers,rangers*2):
    if (p1[i][0] > max(threshold1, 0.7)):
        acc_p1 += 1
        tn[0][0] += 1
    else:
        fn[0][0] += 1
    if (p2[i][0] > max(threshold1, 0.7)):
        acc_p2 += 1
        tn[0][1] += 1
    else:
        fn[0][1] += 1
    if (p3[i][0] > max(threshold1, 0.7)):
        acc_p3 += 1
        tn[0][2] += 1
    else:
        fn[0][2] += 1
    if (p4[i][0] > max(threshold1, 0.7)):
        acc_p4 += 1
        tn[0][3] += 1
    else:
        fn[0][3] += 1

for i in range(0,rangers):
    if (p1[i][1] > max(threshold1-0.5, 0.7)):
        acc2_p1 += 1
        tp[1][0] += 1
    else:
        fp[1][0] += 1
    if (p2[i][1] > max(threshold1-0.5, 0.7)):
        acc2_p2 += 1
        tp[1][1] += 1
    else:
        fp[1][1] += 1
    if (p3[i][1] > max(threshold1-0.5, 0.7)):
        acc2_p3 += 1
        tp[1][2] += 1
    else:
        fp[1][2] += 1
    if (p4[i][1] > max(threshold1-0.5, 0.7)):
        acc2_p4 += 1
        tp[1][3] += 1
    else:
        fp[1][3] += 1


for i in range(rangers,rangers*2):
    if (p1[i][0] > max(threshold1-0.5, 0.7)):
        acc2_p1 += 1
        tn[1][0] += 1
    else:
        fn[0][0] += 1
    if (p2[i][0] > max(threshold1-0.5, 0.7)):
        acc2_p2 += 1
        tn[1][1] += 1
    else:
        fn[1][1] += 1
    if (p3[i][0] > max(threshold1-0.5, 0.7)):
        acc2_p3 += 1
        tn[1][2] += 1
    else:
        fn[1][2] += 1
    if (p4[i][0] > max(threshold1-0.5, 0.7)):
        acc2_p4 += 1
        tn[1][3] += 1
    else:
        fn[1][3] += 1

print "Writing accuracy results for MLP..."

#print(p1)

print("With overconfidence (Threshold around 0.85) " + str(float(acc_p1) / ( (rangers*2))))
print(acc_p1)

print("With lowconfidence (Threshold around 0.7) " + str(float(acc2_p1) / (rangers*2)))

print("Sklearn default score" + str(model1.score(test_data_features, labels)))

print()

print "Writing accuracy results for SGD..."

#print(p2)

print("With overconfidence (Threshold around 0.85) " + str(float(acc_p2) / (rangers*2)))

print("With lowconfidence (Threshold around 0.7) " + str(float(acc2_p2) / (rangers*2)))

print("Sklearn default score" + str(model2.score(test_data_features, labels)))

print()

print "Writing accuracy results for SVM..."

#print(p3)

print("With overconfidence (Threshold around 0.85) " + str(float(acc_p3) / (rangers*2)))

print("With lowconfidence (Threshold around 0.7) " + str(float(acc2_p3) /  (rangers*2)))

result3 = model3.predict(test_data_features)

print("Sklearn default score" + str(model3.score(test_data_features, labels)))

print()

print "Writing accuracy results for Decision Tree..."

#print(p4)

print("With overconfidence (Threshold around 0.85) " + str(float(acc_p4) /  (rangers*2)))

print("With lowconfidence (Threshold around 0.7) " + str(float(acc2_p4) /  (rangers*2)))

print("Sklearn default score" + str(model4.score(test_data_features, labels)))

print()

prec1 = float(tp[0][0]) / (tp[0][0] + fp[0][0])
print("Precision of MLP (Neural Network) with overconfidence" + str(prec1) )
prec2 = float(tp[1][0]) / (tp[1][0] + fp[1][0])
print("Precision of MLP (Neural Network) with low confidence" + str(prec2) )

prec1 = float(tp[0][1]) / (tp[0][1] + fp[0][1])
print("Precision of SGD (Modified Huber) with overconfidence" + str(prec1))
prec2 = float(tp[1][1]) / (tp[1][1] + fp[1][1])
print("Precision of SGD (Modified) with low confidence" + str(prec2) )

prec1 = float(tp[0][2]) / (tp[0][2] + fp[0][2])
print("Precision of SVM (Linear) with overconfidence" + str(prec1) )
prec2 = float(tp[1][2]) / (tp[1][2] + fp[1][2])
print("Precision of SVM (Linear) with low confidence" + str(prec2) )

prec1 = float(tp[0][3]) / (tp[0][3] + fp[0][3])
print("Precision of decision tree with overconfidence" + str(prec1) )
prec2 = float(tp[1][3]) / (tp[1][3] + fp[1][3])
print("Precision of decision tree with low confidence" + str(prec2) )


prec1 = float(tp[0][0]) / (tp[0][0] + fn[0][0])
print("Recall of MLP (Neural Network) with overconfidence" + str(prec1) )
prec2 = float(tp[1][0]) / (tp[1][0] + fn[1][0])
print("Recall of MLP (Neural Network) with low confidence" + str(prec2) )

prec1 = float(tp[0][1]) / (tp[0][1] + fn[0][1])
print("Recall of SGD (Modified Huber) with overconfidence" + str(prec1))
prec2 = float(tp[1][1]) / (tp[1][1] + fn[1][1])
print("Recall of SGD (Modified) with low confidence" + str(prec2) )

prec1 = float(tp[0][2]) / (tp[0][2] + fn[0][2])
print("Recall of SVM (Linear) with overconfidence" + str(prec1) )
prec2 = float(tp[1][2]) / (tp[1][2] + fn[1][2])
print("Recall of SVM (Linear) with low confidence" + str(prec2) )

prec1 = float(tp[0][3]) / (tp[0][3] + fn[0][3])
print("Recall of decision tree with overconfidence" + str(prec1) )
prec2 = float(tp[1][3]) / (tp[1][3] + fn[1][3])
print("Recall of decision tree with low confidence" + str(prec2) )



# #
# # type(sentences)
# # print(sentences[0])
# #
# #
# #
# # import pickle
# # f = open('/media/tugrulz/Yeni Birim/mldata/tfidf.pkl', 'wb')
# # pickle.dump(model, f)
# # f.close()
# #
# #
