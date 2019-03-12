# -*- coding: utf-8 -*
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

n_components = 20
n_top_words = 10

news_file = open('./news.txt', 'r')
lines = news_file.readlines()

data_samples = []
for line in lines:
    if line != "":
        words = line.strip().split(' ')
        temp = []
        for word in words:
            if word.isalpha():
                word = word.lower()
                temp.append(word)
        data_samples.append(" ".join(temp))

swfile = open('./ENstopwords.txt', 'r')
sws = swfile.read()
sws = sws.splitlines()
stop_words = []
for word in sws:
    stop_words.append(word)

tf_vectorizer = CountVectorizer(stop_words=stop_words)
tf = tf_vectorizer.fit_transform(data_samples)

lda = LatentDirichletAllocation(n_components=n_components,
                                max_iter=100,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)

lda.fit(tf)
components = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis]
f = open('./sk_results_20.txt', 'w')
tf_feature_names = tf_vectorizer.get_feature_names()
for topic_idx, topic in enumerate(components):
    f.write("Topic #%d: " % topic_idx + '\n')

    pro_words = topic
    import numpy as np
    index = np.argsort(topic)

    for i in range(len(index)-1, len(index)-1-n_top_words, -1):
        f.write("\t" + tf_feature_names[index[i]] + ", " + str(topic[index[i]]) + '\n')

