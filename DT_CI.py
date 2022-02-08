# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 10:19:49 2020

@author: Morgan Byers
"""

from configparser import ConfigParser
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
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

stats = []

for i in range(10): 
    #shuffle
    X_best, y = shuffle(X_best, y)
    
    #train
    tree = DecisionTreeClassifier(max_depth=100, class_weight='balanced')
    scores = cross_val_score(tree, X_best, y, cv=KFold(n_splits=5))
    
    #evaluate
    stats.append(scores)
    

#get confidence interval for each run
lowers = []
uppers = []

for i, stat in enumerate(stats):
    #gather confidence intervals
    alpha = 0.95
    p = ((1-alpha)/2.0) * 100
    lower = max(0, np.percentile(stat, p))
    p = (alpha+((1-alpha)/2.0)) * 100
    upper = min(1, np.percentile(stat, p))
    
    percent = alpha * 100
    lower_ci = lower * 100
    lowers.append(lower)
    upper_ci = upper * 100
    uppers.append(upper)
    
    print(f'Run {i+1} {percent:.0f}% confidence interval: {lower_ci:.2f}% and {upper_ci:.2f}%')

avg_lower = np.mean(lowers) * 100
avg_upper = np.mean(uppers) * 100
print(f'Overall 95% CI: {avg_lower:.2f}% and {avg_upper:.2f}%')

scores = pd.DataFrame(stats)
scores.to_csv('tree_ci.csv')

#plot accuracies
plt.boxplot(stats)
plt.title('95% CI for 5-fold cross validation')
plt.xlabel('Run')
plt.ylabel('Accuracy')
plt.savefig('tree10_95CI.pdf')
plt.show()


