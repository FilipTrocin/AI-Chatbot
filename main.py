import numpy as np
import tensorflow as tf
import nltk
from nltk.stem.lancaster import LancasterStemmer as stemmer
import tflearn
import random
import os
import json

with open('dialog.json') as file:
    data = json.load(file)


words = []
labels = []
user_input = []
words_labels = []

for dialog in data['dialog']:
    for pattern in dialog['patterns']:
        words_list = nltk.word_tokenize(pattern)  # Separating each word and splitting symbols from them
        words.extend(words_list)
        user_input.append(pattern)
        words_labels.append(dialog['tag'])  # Classifying each word

    if dialog['tag'] not in labels:
        labels.append(dialog['tag'])

