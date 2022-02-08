# -*- coding: utf-8 -*-
"""5 fold cross validation experiment using a keras embedding layer and 
    attention mechanism. 
    
Creates a RNN classifier model that uses Keras' embedding layer and an 
attention mechanism. Creates a classification report and confusion matrix.
    
Created on Fri Jul 17 14:35:25 2020

@author: Morgan "Byers
"""

from configparser import ConfigParser
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.utils import class_weight, shuffle
from keras.layers import Dense, Embedding, Dropout
from keras.layers import Bidirectional
from keras.layers import LSTM
from keras.models import Sequential
from keras.engine.topology import Layer
from keras.utils import np_utils
from keras import initializers, regularizers, constraints
import keras.backend as K
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
import ordinal_categorical_crossentropy as OCC


def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)
    

class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(name='{}_W'.format(self.name),
                                 shape=(input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(name='{}_b'.format(self.name),
                                     shape=(input_shape[-1],),
                                     initializer='zero',
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight(name='{}_u'.format(self.name),
                                 shape=(input_shape[-1],),
                                 initializer=self.init,
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


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



def create_model(vocab_size, dim, lstm_units):
    '''Creates a RNN with the given parameters.

    Args:
        vocab_size (int): The number of words in the vocabulary + 1. 
            Note that 0 is a reserved number for the Keras embedding layer
        dim (int): The desired output dimension of the embedding layer. 
            The embedding layer turns each word into a vector of length dim.
        lstm_units (int): The number of cells in the first LSTM layer.

    Returns:
        model (Sequential): The model.

    '''
    model = Sequential()
    model.add(Embedding(vocab_size, dim, mask_zero=True))
    model.add(Dropout(0.4))
    model.add(Bidirectional(LSTM(lstm_units, return_sequences=True)))
    model.add(LSTM(45, return_sequences=True))
    model.add(AttentionWithContext())
    model.add(Dropout(0.4))
    model.add(Dense(20))
    model.add(Dropout(0.4))
    model.add(Dense(4, activation='softmax'))
    
    return model
    
    
def create_confusion_matrix(trues, preds, class_labels):
    '''Creates and displays a confusion matrix.

    Args:
        trues (list): The actual target classes.
        preds (list): The predicted target classes.
        class_labels (list): The names of the classes. The number of 
            classes should equal len(class_labels)

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
    

def plot_graphs(histories, metric):
    '''Plots a line graph of the validation and training metric across 
        each fold.

    Args:
        histories (list of history objects): A list of history objects 
            from k-fold cross validation.
        metric (string): The name of the metric you are plotting. 

    Returns:
        None.

    '''
    colors = ['b', 'g', 'r', 'c', 'y']
    for i, history in enumerate(histories):
        label_txt = f'fold {i+1} {metric}'
        val_label_txt = f'fold {i+1} val_{metric}'
        plt.plot(history.history[metric], label=label_txt, color=colors[i])
        plt.plot(history.history['val_'+metric], label=val_label_txt, 
                 color=colors[i], linestyle='--')
        plt.xlabel("Epochs")
        plt.ylabel(f'{metric}')
        plt.title(f'{metric} over training epochs (word2vec embedding)')
    plt.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), 
               shadow=True)
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

clean_interviews = preprocess_data(interviews)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(clean_interviews)

#create target 
ratings = pd.read_csv(target_path)

#find average ratings and round
floattarget = list(ratings.mean(axis=1))
target = [int(round(num)) for num in floattarget]

#tokenize and encode sentences
encoded_sentences = tokenizer.texts_to_sequences(clean_interviews)

vocab_size = len(tokenizer.word_index)
print(f'Found {vocab_size} unique words.')

#find max sentence length of all the samples for padding
lens = [len(sent) for sent in encoded_sentences]
max_len = max(lens)

#parameters for the model
dimensionality = 100 #dimensionality of word vectors
lstm_units = 70
vocab_size += 1

n_splits = 5

#input must be an array for KFold
X = pad_sequences(encoded_sentences, maxlen=max_len)
y = np_utils.to_categorical(target, num_classes=4)

#lists to track performance accross K folds
histories = list()
accs = list()
losses = list()
probs = list()
preds = list()
trues = list()


for train_idx, test_idx in KFold(n_splits).split(X):
    xtrain, xtest = X[train_idx], X[test_idx]
    
    ytrain, ytest = y[train_idx], y[test_idx]
    
    #shuffle training set
    xtrain, ytrain = shuffle(xtrain, ytrain)
    
    model = create_model(vocab_size, dimensionality, lstm_units)
    
    model.compile(optimizer='adam', loss=OCC.loss, metrics=['accuracy'])

    eps = 5
    class_weights = class_weight.compute_class_weight(class_weight='balanced', 
                                                      classes=np.unique(target), 
                                                      y=np.array(target))

    history = model.fit(xtrain, ytrain, epochs=eps, verbose=0,
                        validation_split=0.2, class_weight = class_weights)
    
    histories.append(history)

    #test the model
    test_loss, test_acc = model.evaluate(xtest, ytest)
    
    losses.append(test_loss)
    accs.append(test_acc)
    
    print('\nEnd of fold')
    print('-' * 50)
    print(f'Test loss: {test_loss:.3f}')
    print(f'Test accuracy: {test_acc:.3f}')
    print('-' * 50)
    
    #get predictions for confusion matrix and classification report
    probs.extend(model.predict(xtest))
    preds.extend(np.argmax(model.predict(xtest), axis=1))
    trues.extend(np.argmax(ytest, axis=1))
    
    del model
    
avg_acc = np.mean(accs)
stdv_acc = np.std(accs)

avg_loss = np.mean(losses)
stdv_loss = np.std(losses)

print('-'*50)
print('-'*50)
print(f'{n_splits}-fold CV results:')
print(f'Accuracy: {avg_acc:.4f} (+/- {stdv_acc:.4f})')
print(f'Loss: {avg_loss:.4f} (+/- {stdv_loss:.4f})')
print('-'*50)
print('-'*50)

plot_graphs(histories, 'accuracy')
plot_graphs(histories, 'loss')

#create confusion matrix with accumulated predictions
class_labels = [0,1,2,3]
create_confusion_matrix(trues, preds, class_labels)

print(classification_report(trues, preds))
print('-'*50)

#create dataframe of target, predicted, and probabilities
val_results = pd.DataFrame()
val_results['Target'] = trues
val_results['Predicted'] = preds
val_results['Probabilities'] = list(probs)

#save to csv
writepath = config_object["writeSettings"]
save_csv_path = writepath["resultpath"]
val_results.to_csv(save_csv_path)        
