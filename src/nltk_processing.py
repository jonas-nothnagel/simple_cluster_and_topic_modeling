# import the necessary libraries 
import nltk 
import string 
import re 
import pandas as pd
import unicodedata
import gensim
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize 
from nltk.stem.porter import PorterStemmer 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

import pickle 
import sys
sys.setrecursionlimit(100000)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def text_lowercase(text): 
    return text.lower()

def remove_n(text):
    t = re.sub('\n', '', text)
    t = re.sub('  ', '', text)
    t = re.sub('\r', '', text)
    t = re.sub('dp', '', text)
    return t

def remove_numbers(text):
    temp_str = text.split()  
    new_string = [] 

    for word in temp_str: 
        if not word.isdigit(): 
            new_string.append(word) 

    temp_str = ' '.join(new_string) 
    return temp_str

    

def strip_accents(text):

    try:
        text = unicode(text, 'utf-8')
    except NameError: # unicode is a default on python 3 
        pass

    text = unicodedata.normalize('NFD', text)\
           .encode('ascii', 'ignore')\
           .decode("utf-8")

    return str(text)


def remove_punctuation(text): 
    translator = str.maketrans('', '', string.punctuation) 
    return text.translate(translator)