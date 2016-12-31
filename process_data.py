import numpy as np
import cPickle
from collections import defaultdict
import sys, re
import pandas as pd
from dias import *
from random import shuffle

# import mltg
# from mltg import *

path = '/media/tugrulz/Yeni Birim/mldata/subtitles/'
def build_data_sub(data_folder, cv=10, clean_string=True):
    """
    Loads data and split into 10 folds.
    """
    revs = []
    pos_file = data_folder[0]
    neg_file = data_folder[1]
    vocab = defaultdict(float)

    subs = ""
    for nametuple in pos_file:
        #get path
        name = nametuple[0]
        if(name[0:4] == "8 1/2"):
            txt = "8 1slash2 (8slash) (1963)"
        else:
            txt = name.replace('/', "slash")
        txt += ".txt"
        sub = path + txt
        print(sub)

        #Remove stop words

        #PCA / LSA

        #sub should be full adress
        #wtf is he doing? extraneous jobs
        try:
            with open(sub, "rb") as f:
                sub = f.read()
                subs += sub
                spit = sub.split()
                words = set(spit)
                for word in words:
                    vocab[word] += 1
            f.close()
        except(IOError):
            print("No subtitle found for " + name)

    datum = {"y": 1,
             "text": subs,
             "num_words": len(subs.split()),
             "split": np.random.randint(0, cv)}
    revs.append(datum)

    subs = ""

    for nametuple in neg_file:
        #get path
        name = nametuple[0]
        if(name[0:4] == "8 1/2"):
            txt = "8 1slash2 (8slash) (1963)"
        else:
            txt = name.replace('/', "slash")
        txt += ".txt"
        sub = path + txt
        print(sub)

        #Remove stop words

        #PCA / LSA

        #sub should be full adress
        #wtf is he doing? extraneous jobs
        try:
            with open(sub, "rb") as f:
                sub = f.read()
                subs += sub
                spit = sub.split()
                words = set(spit)
                for word in words:
                    vocab[word] += 1
            f.close()
        except(IOError):
            print("No subtitle found for " + name)

    datum = {"y": 0,
             "text": subs,
             "num_words": len(subs.split()),
             "split": np.random.randint(0, cv)}
    revs.append(datum)




    return revs, vocab

def build_data_cv(data_folder, cv=10, clean_string=True):
    """
    Loads data and split into 10 folds.
    """
    revs = []
    pos_file = data_folder[0]
    neg_file = data_folder[1]
    vocab = defaultdict(float)
    with open(pos_file, "rb") as f:
        for line in f:
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"y":1,
                      "text": orig_rev,
                      "num_words": len(orig_rev.split()),
                      "split": np.random.randint(0,cv)}
            revs.append(datum)
    with open(neg_file, "rb") as f:
        for line in f:
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"y":0,
                      "text": orig_rev,
                      "num_words": len(orig_rev.split()),
                      "split": np.random.randint(0,cv)}
            revs.append(datum)
    return revs, vocab
    
def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype='float32')            
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)  

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip() if TREC else string.strip().lower()

def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()

if __name__=="__main__":    
    #w2v_file = sys.argv[1]
    w2v_file = '/media/tugrulz/Yeni Birim/GoogleNews-vectors-negative300.bin'

    tag = 'bleak'

    bleak_best = find_titles("bleak", 14)
    bleak_worst = find_worst_titles("bleak", 14)
    shuffle(bleak_best)
    shuffle(bleak_worst)
    bleak_best_train = bleak_best[0:80]
    print(bleak_best_train)
    bleak_worst_train = bleak_worst[0:80]
    print(bleak_worst_train)
    bleak_best_test = bleak_best[-20:]
    bleak_worst_test = bleak_worst[-20:]

    data_folder = [bleak_best_train, bleak_worst_train]
    print "loading data...",
    revs, vocab = build_data_sub(data_folder, cv=10, clean_string=True)
    max_l = np.max(pd.DataFrame(revs)["num_words"])
    print "data loaded!"
    print "number of sentences: " + str(len(revs))
    print "vocab size: " + str(len(vocab))
    print "max sentence length: " + str(max_l)
    print "loading word2vec vectors...",
    w2v = load_bin_vec(w2v_file, vocab)
    print "word2vec loaded!"
    print "num words already in word2vec: " + str(len(w2v))
    add_unknown_words(w2v, vocab)
    W, word_idx_map = get_W(w2v)
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab)
    W2, _ = get_W(rand_vecs)
    cPickle.dump([revs, W, W2, word_idx_map, vocab], open("/media/tugrulz/Yeni Birim/mldata/mr.p", "wb"))
    cPickle.dump([bleak_best_test, bleak_worst_test], open("/media/tugrulz/Yeni Birim/mldata/bleaktest.p", "wb"))
    print "dataset created!"
    
