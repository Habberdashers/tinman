# encoding=utf8  
import sys
import nltk
import random
import codecs
#from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from nltk.tokenize import word_tokenize


bills_passed = open("passed.txt","r").read()
bills_rejected = open("rejected.txt","r").read()


documents = []


for r in bills_passed.split('\n'):
    documents.append( (r, "pos") )


for r in bills_rejected.split('\n'):
    documents.append( (r, "neg") )

all_words = []


bills_passed_words = word_tokenize(bills_passed)
bills_rejected_words = word_tokenize(bills_rejected)


for w in bills_passed_words:
    all_words.append(w.lower())

for w in bills_rejected_words:
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:200]

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

featuresets = [(find_features(rev), category) for (rev, category) in documents]

random.shuffle(featuresets)

# positive data example:      
training_set = featuresets[:100]
testing_set =  featuresets[100:]



classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Naive Bayes accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(10)



