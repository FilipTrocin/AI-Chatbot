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


