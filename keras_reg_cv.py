# -*- coding: utf-8 -*-
"""Runs 5 fold CV on a regression RNN model. 

Creates a classification report and confusion matrix of the model's 
predictions aggregated (and rounded) accross all folds.

Created on Mon Jul 13 13:52:52 2020

@author: Morgan Byers
"""


from configparser import ConfigParser
from keras.layers import Dense, Embedding, LSTM, Dropout, Bidirectional
from keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
import string
import tensorflow as tf
from tensorflow.python.keras.preprocessing.text import Tokenizer


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
    

def create_model(vocab_size, dimensionality, lstm_units):
    """Creates and compiles a RNN with specified parameters.

    Args:
        pretrained_weights (TYPE): The weights learned by the embedding layer.
        vocab_size (int): The size of vocabulary learned by embedding layer.
        dimensionality (int): The dimensionality of the word vectors.
        lstm_units (int): The number of LSTM units.

    Returns:
        model (Sequential): The compiled RNN.

    """
    model = Sequential()
    model.add(Embedding(vocab_size, dimensionality))
    model.add(Dropout(0.4))
    model.add(Bidirectional(LSTM(units=lstm_units, 
                             input_shape=(None, dimensionality), 
                             return_sequences=True)))
    model.add(LSTM(units=20, input_shape=(None, dimensionality)))
    model.add(Dropout(0.4))
    model.add(Dense(units=10))
    model.add(Dropout(0.4))
    model.add(Dense(units=1, activation='relu'))
    optimizer = tf.keras.optimizers.Adam()
    
    model.compile(optimizer=optimizer, loss='mse')
    
    return model


def train(model, xtrain, ytrain):
    '''Trains the model given an array of features and target

    Args:
        model (Sequential): The model to be trained.
        xtrain (numpy array): A collection of training data. 
            Assumed shape is (num_samples, None).
        ytrain (numpy array): A collection of one-hot encoded targets.
            Assumed shape is (num_samples, 1)

    Returns:
        avg_loss (float): The average loss from each instance.

    '''
    losses = list()
    for feature, label in zip(xtrain, ytrain):
        #reshape input data
        timesteps = len(feature)
        feature = np.reshape(feature, (1, timesteps))
        label = np.reshape(label, (1,1))
        #train
        history = model.fit(feature, label, batch_size=1, epochs=1, 
                            verbose=0)
        #record stats
        losses.append(history.history['loss'])
    avg_loss = np.mean(losses)
    return avg_loss


def test(model, xtest, ytest):
    """Evaluate a model on the given test set.

    Args:
        model (Sequential): The model to be evaluated.
        xtest (numpy array): A collection of test data. 
            Assumed shape is (num_samples, None).
        ytest (numpy array): A collection of one-hot encoded targets.
            Assumed shape is (num_samples, 4)

    Returns:
        avg_loss (float): The average loss accross each prediction.

    """
    losses = list()
    for feature, label in zip(xtest, ytest):
        #reshape input data
        timesteps = len(feature)
        feature = np.reshape(feature, (1, timesteps))
        label = np.reshape(label, (1,1))
        #test
        loss = model.evaluate(feature, label, verbose=0)
        #record stats
        losses.append(loss)
    avg_loss = np.mean(losses)
    return avg_loss


def get_preds(model, X):
    """Get model predictions on an input.

    Args:
        model (Keras model): The model that will be making predictions.
        X (numpy array): The input for the model to predict on. 
            Assumed shape is (num_samples, None)

    Returns:
        preds (list): The predictions for each instance.

    """
    preds = list()
    for feature in X:
        timesteps = len(feature)
        feature = np.reshape(feature, (1, timesteps))
        prob = model.predict_on_batch(feature)
        preds.append(prob.item())
    return preds
    

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


#read data using config parser
config_object = ConfigParser()
config_object.read("config.ini")

paths = config_object["appSettings"]

data_path = paths["datapath"]
target_path = paths["targetpath"]

writepath = config_object["writeSettings"]
save_csv_path = writepath["resultpath"]

data = pd.read_csv(data_path)

#clean the interviews
interviews_clean = preprocess_data(list(data['sentences_clean'].values))

#remove sentences without ratings
del interviews_clean[1044]
del interviews_clean[314]

#create target 
ratings = pd.read_csv(target_path)

#find average ratings and round, this is the target
target = list(ratings.mean(axis=1))

#tokenize and encode sentences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(interviews_clean)

vocab_size = len(tokenizer.word_index)
print(f'Found {vocab_size} tokens.')

#params for network
dimensionality = 30 #dimensionality of word vectors
lstm_units = 20
vocab_size += 1
    
n_splits = 5

#input must be an array for KFold
X = np.array(tokenizer.texts_to_sequences(interviews_clean))
target = np.array(target)

#lists to track performance accross K folds
all_losses = list()
all_preds = list()
all_floats = list()
all_trues = list()
all_true_floats = list()

for train_idx, test_idx in KFold(n_splits).split(X):
    print('-'*40)
    print('-'*40)
    #to work with functions as-is, convert back to list
    #TODO: find a way to not do this
    ytrain = list(target[train_idx])
    ytest = list(target[test_idx])
    
    #convert encoded sentences into xtrain and xtest arrays
    xtrain = X[train_idx]
    xtest = X[test_idx]
    
    #create model and show summary
    model = create_model(vocab_size, dimensionality, lstm_units)
    
    #track epochs manually
    epochs = 20
    
    #train the model
    print('Training model now...')
    for e in range(1, (epochs+1)):
        xtrain, ytrain = shuffle(xtrain, ytrain)
        print(f'Epoch {e}/{epochs}:')
        avg_loss = train(model, xtrain, ytrain)
        val_loss = test(model, xtest, ytest)
        print(f'\tTrain loss: {avg_loss:.4f}')
        print(f'\tVal loss: {val_loss:.4f}')
        
    #test the model
    print('-'*40)
    test_loss = test(model, xtest, ytest)
    print(f'Test loss: {test_loss:.3f}')
    
    all_losses.append(test_loss)
    
    #get predictions 
    preds = get_preds(model, xtest)
    trues = ytest
    all_floats.extend(preds)
    all_true_floats.extend(trues)
    
    #round them for classification report and confusion matrix
    round_preds = [int(round(num)) for num in preds]
    true_ints = [int(round(num)) for num in trues]
    all_trues.extend(true_ints)
    all_preds.extend(round_preds)
    
    
    del model
    

#print average loss and accuracy accross all folds
avg_loss = np.mean(all_losses)
loss_stdv = np.std(all_losses)

print(f'Average {n_splits}-fold CV MSE: {avg_loss:.3f} +/-({loss_stdv:.3f})')


#create confusion matrix with preds accross all folds
class_labels = [0,1,2,3]
create_confusion_matrix(all_trues, all_preds, class_labels)

print(classification_report(all_trues, all_preds))

#create dataframe of target, predicted, and probabilities
val_results = pd.DataFrame()
val_results['Target'] = all_true_floats
val_results['Rounded target'] = all_trues
val_results['Predicted'] = all_floats
val_results['Rounded prediction'] = all_preds

#save to csv
val_results.to_csv(save_csv_path)