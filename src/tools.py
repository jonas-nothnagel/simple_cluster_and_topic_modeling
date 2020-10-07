#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 15:46:44 2020

@authors: Mykola Skrynnik &  Jason Liang & Jonas Nothnagel

@title: helper functions for cleaning and regressor extraction
"""

from typing import Iterable, Tuple

import numpy as np

import pandas as pd

from itertools import chain


import matplotlib.pyplot as plts

import seaborn as sns; sns.set()

import spacy

spacy.prefer_gpu() # If CUDA-compatible GPU is available

import en_core_web_sm

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])
 
from sklearn.cluster import KMeans 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances
import matplotlib.cm as cm
import seaborn as sns
import matplotlib.pyplot as plt

from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import unicodedata
import re


import pycountry

def spacy_clean(alpha:str, use_nlp:bool = False) -> str:

    """

    Clean and tokenise a string using Spacy. Keeps only alphabetic characters, removes stopwords and

    filters out all but proper nouns, nounts, verbs and adjectives.

    

    Parameters

    ----------

    alpha : str

            The input string.

    use_nlp : bool, default False

            Indicates whether Spacy needs to use NLP. Enable this when using this function on its own.

            Should be set to False if used inside nlp.pipeline

            

     Returns

    -------

    ' '.join(beta) : a concatenated list of lemmatised tokens, i.e. a processed string

    

    Notes

    -----

    Fails if alpha is an NA value. Performance decreases as len(alpha) gets large.

    Use together with nlp.pipeline for batch processing.



    """

    

    if use_nlp:

        alpha = nlp(alpha)

        

    beta = []

    for tok in alpha:

        if all([tok.is_alpha, not tok.is_stop, tok.pos_ in ['PROPN', 'NOUN', 'VERB', 'ADJ']]):

            beta.append(tok.lemma_)

            

    return ' '.join(beta)

#data processing
def process_data(data):
    
    data['clean_text'] = [spacy_clean(batch) for batch in tqdm(nlp.pipe(data['text'], batch_size = 10))]
    
    #check if some documents were not parsed:
    checked = data[data.clean_text != ""]
    if len(checked) != len(data):
        print('deleted empty texts')
    incorrect_index = []
    for i, row in checked.iterrows():
        if len(row.clean_text) < 1000:
            print('country name with faulty document:', checked.country.iloc[i])
            incorrect_index.append(i)
    checked = checked.drop(incorrect_index)
    
    #lowercase
    checked['clean_text'] = checked['clean_text'].apply(lambda x:x.lower())

    return checked

#tokenizer
def tokenize(text):
    return [tok.text for tok in nlp.tokenizer(text)]

#Sklearn for feature extraction
def feature_extraction(text, min_ngram, max_ngram, min_df_value, max_df_value):
    
    clean_text = text
    tf_idf_vectorizor = TfidfVectorizer(ngram_range = (min_ngram,max_ngram), min_df = min_df_value, max_df = max_df_value)

    vec = tf_idf_vectorizor.fit(clean_text)
    tf_idf = vec.transform(clean_text)
    tf_idf_norm = normalize(tf_idf)
    tf_idf_array = tf_idf_norm.toarray()
    #look at features
    pd.DataFrame(tf_idf_array, columns=tf_idf_vectorizor.get_feature_names()).head()
    
    return tf_idf_array, tf_idf_vectorizor, vec


#PCA for dimension reduction:
def pca_d_reduction(vector, components):
    sklearn_pca = PCA(n_components = components)
    pca = sklearn_pca.fit(vector)
    print('number of components:',pca.n_components_)
    Y_sklearn = sklearn_pca.fit_transform(vector)
    
    return Y_sklearn, pca

def get_top_features_cluster(tf_idf_array, prediction, n_feats, vetorizer):
    labels = np.unique(prediction)
    dfs = []
    for label in labels:
        id_temp = np.where(prediction==label) # indices for each cluster
        x_means = np.mean(tf_idf_array[id_temp], axis = 0) # returns average score across cluster
        sorted_means = np.argsort(x_means)[::-1][:n_feats] # indices with top 20 scores
        features = vetorizer.get_feature_names()
        best_features = [(features[i], x_means[i]) for i in sorted_means]
        df = pd.DataFrame(best_features, columns = ['features', 'score'])
        dfs.append(df)
    return dfs

def infer_query(query, vec, pca, kmeans, dfs, dictlist, dataframe):
    
    query = [query]
    query_vec = vec.transform(query)
    query_vec_norm = normalize(query_vec)
    query_vec_array = query_vec_norm.toarray()
    query_vec_pca = pca.transform(query_vec_array)

    infer = kmeans.predict(query_vec_pca)
    number = int(infer)
    plt.figure(figsize=(6,4))
    sns.barplot(x = 'score' , y = 'features', orient = 'h' , data = dfs[number][:7])

    projects = dictlist[number]
    project_id = []
    for i in projects[1]:
        project_id.append(dataframe.country.iloc[i])

    print('Query:', query)
    print('')
    print('Countries that are part of this cluster:')
    print(project_id)
    print('')
    print('top words of predicted cluster:')
    
    
def remove_stopwords(text): 
    stop_words = set(stopwords.words("english")) 
    country_list = []
    for country in pycountry.countries:
        country_list.append(country.name)
    countries_lower = [item.lower() for item in country_list]
    stop_words_custom =  ['a', 'according', 'achieve', 'achieve sustainable', 'action', 
                  'b', 'c', 'd','e','f','g','h','i','j','k','l','m','n','o','p','q',
                  'r','s','t','u','v','w','x','y','z', 'added', 'addition', 'address',
             'affect', 'ages', 'ages persons', 'ages year', 'iii','executive','board',
             'distr','general','english','session','annex','iv', 'content','page',
             'ages year older', 'agreed', 'aids', 'aiming', 'document','draft',
           'allocated', 'appropriate', 'areas', 'assistance', 
           'associated', 'attention', 'availability', 'based', 'first','second',
           'original','rationale','chapter','basis', 'cent', 
           'circumstances','almost','least','sdgs','new york', 'include',
       'contribution', 'covered', 'coverage', 'currently', 'double', 'baseline', 'capacity', 'government', 'policy', 'system', 'reports', 
         'effective', 'ensure', 'forms', 'global', 'relate','item','service', 
              'ii', 'increase', 'increase number', 'country','strengthen', 'sub', 'na', 
                          'sp', 'au', 'nacional', 'de', 
              'months', 'national', 'number', 'particular',
                    'target', 'dp', 'dpc', 'indicator', 
           'persons', 'persons attributed', 'persons directly', 'program','matter', 
          'ratio', 'project', 'office',
         'states', 'states development', 'states' ,'development', 'countries', 
           'states dollars', 'support',  'january','february','march','april','may','june','july','august','september',
           'october','november','december','year', 'year ages', 'new','york',
            'undp', 'united','programme', 'nations', 'irrf' 'de', 'el', 'swaziland', 'viet', 'nam', 'irrf', 'bolivia', 'philippine', 'saudi', 'arabia', 'salvador']
    
#'policiy', 'system','report', 'implementation', 'implement', 'improve', 'yearly', 'federal', 'district', 'provincial', 'percent', 'governorate', 'seventh', 

    newstop_words = stop_words_custom + countries_lower
    stop_tokens = []
    for i in newstop_words:
        tok = word_tokenize(i) 
        stop_tokens.append(tok)
    y = list(chain(*stop_tokens))

    stop_words.update(y)
    word_tokens = word_tokenize(text) 
    filtered_text = [word for word in word_tokens if word not in stop_words] 
    return ' '.join(filtered_text)

def strip_accents(text):

    try:
        text = unicode(text, 'utf-8')
    except NameError: # unicode is a default on python 3 
        pass

    text = unicodedata.normalize('NFD', text)\
           .encode('ascii', 'ignore')\
           .decode("utf-8")

    return str(text)

def remove_dp(text):   
    t=re.sub(r"\bdp\w+", "", text)
    return t
