import pandas as pd
import numpy as np

import nltk.stem
import lda
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

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
class StemmedCountVectorizer(CountVectorizer):
  def build_analyzer(self):
    analyzer = super(StemmedCountVectorizer, self).build_analyzer()
    return lambda doc: ([english_stemmer.stem(w) for w in analyzer(doc)])

vectorizer = StemmedCountVectorizer(analyzer='word',   \
                             ngram_range=(1,1), \
                             min_df=0,          \
                             stop_words='english')

train_data_features = vectorizer.fit_transform(corpus)
feature_names = vectorizer.get_feature_names()
feature_names_array = np.array(feature_names)

no_topics = 20

lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(train_data_features)

no_top_words = 10
display_topics(lda, feature_names_array, no_top_words)
