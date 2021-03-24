# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 12:55:49 2021

@author: Divyasha Pradhan
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


df = pd.read_csv('balanced_reviews.csv')

df.head()

df['reviewText'].head()

df['overall'].value_counts()

df.isnull().any(axis = 0)

df.dropna (inplace = True)

df['overall'] != 3

df = df[df['overall'] != 3]

df['overall'].value_counts()


df['Positivity'] = np.where (df['overall'] > 3, 1, 0)

df['Positivity'].value_counts()

#reviewText - features
#Positivity - labels


features_train, features_test, labels_train, labels_test  = train_test_split(df['reviewText'], df['Positivity'], random_state = 42)

from sklearn.feature_extraction.text import CountVectorizer


vect = CountVectorizer().fit(features_train)
len(vect.get_feature_names())

vect.get_feature_names()[10000:10010]


features_train_vectorized = vect.transform(features_train)
#features_train_vectorized.toarray()

#model 01
model = LogisticRegression()

model.fit(features_train_vectorized, labels_train)

predictions = model.predict(vect.transform(features_test))

from sklearn.metrics import confusion_matrix


confusion_matrix(labels_test, predictions)


from sklearn.metrics import roc_auc_score

roc_auc_score(labels_test, predictions)

#saving the pickle files

pkl_filename = "pickle_model.pkl"
vocab_filename = "feature.pkl"



import pickle

with open(pkl_filename, 'wb') as file:
    pickle.dump(model, file)
with open(vocab_filename, 'wb') as file:
    pickle.dump(vect.vocabulary_, file)
