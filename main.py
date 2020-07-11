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


words = []  # ['hi', 'hello','good', 'morn', 'is', 'anyon', 'ther', ...]
labels = []  # ["abilities", "age", "experience", "farewell", "greeting", ...]
user_input = []  # ['Hi', 'Hello', 'Hey', 'Yo', 'See you', 'cya', 'Goodbye', ...]
words_labels = []  # ['greeting', 'greeting', 'greeting', 'greeting', 'farewell', 'farewell', 'farewell', ...]

for dialog in data['dialog']:
    for pattern in dialog['patterns']:
        words_list = nltk.word_tokenize(pattern)  # Separating each word and splitting symbols from them
        stemmed_tokenized = [stemmer.stem(wrd.lower()) for wrd in words_list]  # Removing prefix from every word in list
        words.extend(stemmed_tokenized)
        stemmed_ui = [stemmer.stem(ptrn.lower()) for ptrn in pattern.split()]
        user_input.extend(stemmed_ui)
        words_labels.append(dialog['tag'])  # Classifying each word

    if dialog['tag'] not in labels:
        labels.append(dialog['tag'])

words = sorted(list(set(words)))  # removing duplicates from the list, sort elements in certain order
labels = sorted(labels)

training = []
outcome = []

empty_list = [0 for _ in range(len(labels))]
