# Import the built-in logging module and configure it so that Word2Vec
# creates nice output messages
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

# Set values for various parameters
num_features = 20    # Word vector dimensionality
min_word_count = 100   # Minimum word count
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words

path = "/media/tugrulz/Yeni Birim/mldata/OpenSubtitles2016.raw.en/OpenSubtitles2016.raw.en"
saved = "/media/tugrulz/Yeni Birim/mldata/"

# sentences = []
#
# with open(path, 'rb') as f:
#     sentences = map(rstrip, f)



# Initialize and train the model (this will take some time)
from gensim.models import word2vec
from gensim.models.word2vec import LineSentence

sentences = LineSentence(path)

print "Training model..."
model = word2vec.Word2Vec(sentences, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)

# If you don't plan to train the model any further, calling
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

# It can be helpful to create a meaningful model name and
# save the model for later use. You can load it later using Word2Vec.load()
model_name = "300features_40minwords_10context"
model.save(saved+model_name)