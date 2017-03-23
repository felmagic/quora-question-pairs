import pandas as pd
import numpy as np

import nltk.stem
import lda
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

def display_topics(model, feature_names, no_top_words):
  for topic_idx, topic in enumerate(model.components_):
    print "Topic %d:" % (topic_idx)
    print " ".join([feature_names[i]
      for i in topic.argsort()[:-no_top_words - 1:-1]])

train = pd.read_csv("train.csv", header=0)
train_size = train.shape[0]
corpus = []

for i in xrange(0, train_size):
  if((i+1) % 1000 == 0):
    print "questions %d of %d\n" % (i + 1, train_size)
  if type(train["question1"][i]) is str:
    corpus.append(unicode(train["question1"][i], "utf-8"))
  if type(train["question2"][i]) is str:
    corpus.append(unicode(train["question2"][i], "utf-8"))

english_stemmer = nltk.stem.SnowballStemmer('english')
class StemmedTfidfVectorizer(TfidfVectorizer):
  def build_analyzer(self):
    analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
    return lambda doc: ([english_stemmer.stem(w) for w in analyzer(doc)])

vectorizer = StemmedTfidfVectorizer(max_df=0.95, \
                                    min_df=2, \
                                    max_features=1000, \
                                    stop_words='english')

train_data_features = vectorizer.fit_transform(corpus)
feature_names = vectorizer.get_feature_names()
feature_names_array = np.array(feature_names)

no_topics = 20
no_top_words = 10

nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(train_data_features)
display_topics(nmf, feature_names_array, no_top_words)
