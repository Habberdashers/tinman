# encoding=utf8 
from flask import Flask, request, jsonify, json
import sys
import nltk
import random
import codecs
#from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


app = Flask(__name__, static_url_path='')
app.debug = True





@app.route("/hello", methods=['GET'])
def say_hi():
	return "Hello Tin Man"


@app.route('/test/<send_the_rule>', methods=['POST'])
def check_rule(send_the_rule):
  if request.method == 'POST' or send_the_rule:
  	bills_passed = open("static/bills_passed.txt","r").read()
  	bills_rejected = open("static/bills_rejected.txt","r").read()

 
  	documents = []
	for r in bills_passed.split('\n'):
		documents.append( (r, "liberal") )

	for r in bills_rejected.split('\n'):
		documents.append( (r, "conservative") )



	all_words = []

	bills_passed_words = word_tokenize(bills_passed)
	#bills_rejected_words = word_tokenize(bills_rejected)

	for w in bills_passed_words:
		all_words.append(w.lower())

	"""
	for w in bills_rejected_words:
		all_words.append(w.lower())
	"""


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

	print (send_the_rule)
	# positive data example:      
	training_set = featuresets[:1000]
	testing_set = featuresets[1000:]
	#testing_set = find_features('Gun Control')

	classifier = nltk.NaiveBayesClassifier.train(training_set)
	print (classifier.show_most_informative_features(5))
  	return jsonify({'status': 201, 'error': None, 'payload': {'response': 'testing', 'Naive Bayes accuracy_percent': (nltk.classify.accuracy(classifier, testing_set))*100}})



@app.route("/predict", methods=['POST'])
def start_training():
	bills_passed = open("static/bills_passed.txt","r").read()
	bills_rejected = open("static/bills_rejected.txt","r").read()

	documents = []
	for r in bills_passed.split('\n'):
		documents.append( (r, "liberal") )

	for r in bills_rejected.split('\n'):
		documents.append( (r, "conservative") )

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
	training_set = featuresets[:1000]
	testing_set = featuresets[1000:]


	classifier = nltk.NaiveBayesClassifier.train(training_set)
	print("Naive Bayes accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)
	#classifier.show_most_informative_features(1)

	#print (classifier.show_most_informative_features(5))
	return jsonify({'status': 201, 'error': None, 'payload': {'response': classifier.show_most_informative_features(6), 'accuracy_percent': (nltk.classify.accuracy(classifier, testing_set))*100}})

if __name__ == "__main__":
	app.run()
