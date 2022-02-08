# -*- coding: utf-8 -*-
"""Performs cross validation and parameter tuning on logistic regression.

Created on Thu Jul 30 09:02:39 2020

@author: Morgan Byers
"""

from configparser import ConfigParser
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
import string


def preprocess_data(interviews):
    '''Cleans the given data by removing numbers and punctuation. Does not 
    tokenize the sentences.

    Args:
        interviews (list): The corpus to be cleaned.

    Returns:
        interviews (list): The cleaned corpus.

    '''
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    clean_corpus = list()
    for sentence in interviews:
        words = list()
        for word in sentence.split():
            words.append(word.lower())
        words_joined = ' '.join(words)
        clean_strings = regex.sub('', words_joined)
        clean_corpus.append(clean_strings)
    
    return clean_corpus


def create_confusion_matrix(trues, preds, class_labels):
    '''Generates and displays a confusion matrix.

    Args:
        trues (list): The true value of each target.
        preds (list): The predicted value of each target.
        class_labels (list): The class labels. Each class label must show 
            up at least once in trues.

    Returns:
        None.

    '''
    cm = confusion_matrix(trues, preds, labels=class_labels)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('True labels')
    plt.xlabel('Predicted labels')
    plt.show()


#read data using config parser
config_object = ConfigParser()
config_object.read("config.ini")

paths = config_object["appSettings"]
data_path = paths["datapath"]
target_path = paths["targetpath"]

data = pd.read_csv(data_path)

#clean the interviews
interviews = list(data['sentences_clean'].values)

#remove sentences without ratings
del interviews[1044]
del interviews[314]

#create target 
ratings = pd.read_csv(target_path)

#find average ratings and round
floattarget = list(ratings.mean(axis=1))
y = [int(round(num)) for num in floattarget]

#clean sentences
clean_data = preprocess_data(interviews)

#vectorize sentences into BOW
vectorizer = CountVectorizer(clean_data, ngram_range=(1,3))
X = vectorizer.fit_transform(clean_data)

X_best = SelectKBest(chi2, k=12500).fit_transform(X, y)

all_acc = list()
all_preds = list()
all_target = list()

for train_i, test_i in KFold(n_splits=10).split(X_best):
    xtrain, xtest = X_best[train_i], X_best[test_i]
    
    #y must be an array
    y = np.array(y)
    ytrain, ytest = y[train_i], y[test_i]
    
    #shuffle training data
    xtrain, ytrain = shuffle(xtrain, ytrain)
    
    #create model
    lr = LogisticRegression(C=10, class_weight='balanced', max_iter=600)
    
    lr.fit(xtrain, ytrain)
    
    #evaluate performance and get predictions
    acc = lr.score(xtest, ytest)
    all_acc.append(acc)
    
    preds = lr.predict(xtest)
    all_preds.extend(preds)
    
    all_target.extend(ytest)
    
    
#get dataframe of all the accuracies from each fold and display
fold_accs = pd.DataFrame(all_acc, columns=['Test accuracy'])
print('Test accuracy across each fold: ')
print(fold_accs.head(10))
    
#display results accross all folds
avg_acc = np.mean(all_acc)
std_acc = np.std(all_acc)

print()
print('-'*50)
print(f'Average 10-fold CV accuracy: {avg_acc:.3f} (+/- {std_acc:.3f})')
print('-'*50)
print()

#create confusion matrix and classification report accross all folds
create_confusion_matrix(all_target, all_preds, [0,1,2,3])

print(classification_report(all_target, all_preds))
