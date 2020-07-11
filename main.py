import numpy as np
import tensorflow as tf
import nltk
from nltk.stem.lancaster import LancasterStemmer
import tflearn
import random
import os
import json

stemmer = LancasterStemmer()

with open('dialog.json') as file:
    data = json.load(file)


words = []
labels = []
user_input = []
words_labels = []

for dialog in data['dialog']:
    for pattern in dialog['patterns']:
        words_list = nltk.word_tokenize(pattern)  # Separating each word and splitting symbols from them
        stemmed = [stemmer.stem(wrd.lower()) for wrd in words_list]  # Removing prefix from every word in list
        words.extend(stemmed)
        user_input.append(pattern)
        words_labels.append(dialog['tag'])  # Classifying each word

    if dialog['tag'] not in labels:
        labels.append(dialog['tag'])

words = sorted(list(set(words)))  # removing duplicates from the list, sort elements in certain order

labels = sorted(labels)


