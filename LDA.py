# -*- coding: utf-8 -*-
"""Performs topic modeling on the given corpus.

This experiment uses sklearn's implementation of Latent Dirichlet Allocation. 
A grid search CV experiment is used to find the optimal number of topics and 
learning rate. Then, results are visualized and saved in an html file.

Created on Wed Jun 24 13:29:15 2020

@author: Morgan Byers
"""


from configparser import ConfigParser
import gensim
import matplotlib.pyplot as plt
import pandas as pd
import pyLDAvis
import pyLDAvis.sklearn
import re
import seaborn as sns
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV


REPLACE_NUMS = re.compile("[^0-9a-z #+_]")

def preprocess_data(interviews):
    interviews = [REPLACE_NUMS.sub("", line.lower()) for line in interviews]
    return interviews


def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True)) 
        

#read data using config parser
config_object = ConfigParser()
config_object.read("config.ini")

paths = config_object["appSettings"]
data_path = paths["datapath"]

#also get write paths with config parser
writepaths = config_object['writeSettings']
lda_path = writepaths['ldapath']
tuned_lda_path = writepaths['tunedldapath']

df = pd.read_csv(data_path)

#clean sentences and tokenize
interviews_clean = preprocess_data(list(df['sentences_clean'].values)) 
data_words = list(sent_to_words(interviews_clean))

prepared_texts = list()
for sent in data_words:
  prepared_texts.append(" ".join([token for token in sent]))

#vectorize
vectorizer = CountVectorizer(analyzer='word', min_df=1, ngram_range=(1,2), 
                             stop_words='english', lowercase=True, 
                             token_pattern='[a-zA-Z]{2,}')

data_vectorized = vectorizer.fit_transform(prepared_texts)

#create model
lda_model = LatentDirichletAllocation(n_components=5, max_iter=5, 
                                      learning_method='online', 
                                      learning_decay=0.7, random_state=100,
                                      batch_size=128, evaluate_every=-1,
                                      n_jobs = -1)

lda_output = lda_model.fit_transform(data_vectorized)

#create and save topic model visualization
pyLDAvis.enable_notebook()
panel = pyLDAvis.sklearn.prepare(lda_model, data_vectorized, 
                                 vectorizer, mds='tsne')

pyLDAvis.save_html(panel, lda_path)

#params for grid search
search_params = {'n_components': [5, 10, 20, 15], 
                 'learning_decay': [.3, .5, .7]
                 }

#new model
lda = LatentDirichletAllocation()
 
#grid search
model = GridSearchCV(lda, param_grid=search_params)
model.fit(data_vectorized)

#display results
best_lda_model = model.best_estimator_

print("Best Model's Params: ", model.best_params_)
print("Best Log Likelihood Score: ", model.best_score_)
print("Model Perplexity: ", best_lda_model.perplexity(data_vectorized))

#visualize
results = pd.DataFrame(model.cv_results_)
current_palette = sns.color_palette("Set2", 3)
plt.figure(figsize=(12,8))

sns.lineplot(data=results, x='param_n_components', y='mean_test_score',
             hue='param_learning_decay', palette=current_palette, marker='o')
plt.show()

#create notebook with tuned visuals
panel = pyLDAvis.sklearn.prepare(best_lda_model, data_vectorized, 
                                 vectorizer, mds='tsne')

pyLDAvis.save_html(panel, tuned_lda_path)