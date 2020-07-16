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

dictionary = []  # ['anyon', 'good', 'hello', 'hi', 'is', 'morn', 'ther', ...]
labels = []  # ["abilities", "age", "experience", "farewell", "greeting", ...]
words = []  # ['hi', 'hello', 'hey', 'yo', 'see you', 'cya', 'goodby', ...]
words_labels = []  # ['greeting', 'greeting', 'greeting', 'greeting', 'farewell', 'farewell', 'farewell', ...]


def extracter(dict, words, wrds_labels, wrd_class):
    """
    Take pattern from 'dialog' in .json file, remove redundant symbols and prefixes from these words, assign an affiliation
    tag for each word, create a list of unique labels
    """
    for dialog in data['dialog']:
        for pattern in dialog['patterns']:
            words_list = nltk.word_tokenize(pattern)  # Separating each word and splitting symbols from them
            stemmed_tokenized = [stemmer.stem(wrd.lower()) for wrd in
                                 words_list]  # Removing prefix from every word in list
            dict.extend(stemmed_tokenized)

            # Difference between len of dictionary and len of words labels if from not using tokenizer
            # Tokenizer removes 'm and 's while .split does not do that
            stemmed_ui = [stemmer.stem(ptrn.lower()) for ptrn in pattern.split()]
            words.extend(stemmed_ui)
            [wrds_labels.append(dialog['tag']) for _ in stemmed_ui]  # Classifying each word

        if dialog['tag'] not in wrd_class:
            wrd_class.append(dialog['tag'])

    return dict, words, wrds_labels, wrd_class


extracter(dictionary, words, words_labels, labels)

dictionary = sorted(list(set(dictionary)))  # removing duplicates from the list, sort elements in certain order (53 el.)
labels = sorted(labels)

training = []  # bag of words indicating whether the word exist in dictionary or not
outcome = []  # lists indicating affiliation to the label of each word

empty_list = [0 for _ in range(len(labels))]


def segregate(words_list):
    """
    Determining the affiliation of every word
    ":arg: list of all words in patterns in .json file
    """
    for count, word in enumerate(words_list):
        bag = []
        if word in dictionary:
            bag.append(1)
        else:
            bag.append(0)

        output_row = empty_list[:]  # Making copy of the "empty_list"

        # Assigning 1 to the row from which the current word (based on count) comes from
        output_row[labels.index(words_labels[count])] = 1
        training.append(bag)
        outcome.append(output_row)


segregate(words)

training = np.array(training)
outcome = np.array(outcome)

tf.reset_default_graph()  # resetting previous settings

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 10, activation='softmax')
net = tflearn.fully_connected(net, len(outcome[0]), activation='sigmoid')
net = tflearn.regression(net)

model = tflearn.DNN(net)

model.fit(training, outcome, n_epoch=3000, batch_size=6, show_metric=True)

model.save("chatbot_model.tflearn")
