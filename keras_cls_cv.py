# -*- coding: utf-8 -*-
"""Runs 5 fold CV on a RNN for text classification.

Creates a classification report and confusion matrix of the model's 
predictions aggregated accross all folds.

Created on Mon Jul 13 11:29:30 2020

@author: Morgan Byers
"""

from configparser import ConfigParser
from keras.layers import Dense, Embedding, LSTM, Dropout
from keras.models import Sequential
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.utils import shuffle, class_weight
import string
import tensorflow as tf
from tensorflow.python.keras.preprocessing.text import Tokenizer
import ordinal_categorical_crossentropy as OCC


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
    '''Creates a compiled model with one LSTM layer.

    Args:
        vocab_size (int): The size of the vocab.
        dimensionality (int): The output dimension of the embedding layer.
        lstm_units (int): The number of cells in the LSTM layer.

    Returns:
        model (Keras Sequential Model): The compiled model.

    '''
    model = Sequential()
    model.add(Embedding(vocab_size, dimensionality))
    model.add(Dropout(0.4))
    model.add(LSTM(units=lstm_units, input_shape=(None, dimensionality)))
    model.add(Dropout(0.4))
    model.add(Dense(units=10))
    model.add(Dropout(0.4))
    model.add(Dense(units=4, activation='softmax'))
    optimizer = tf.keras.optimizers.Adam()
    
    model.compile(optimizer=optimizer, loss=OCC.loss, metrics=['accuracy'])
    
    return model


def train(model, xtrain, ytrain, class_weights):
    '''Trains the model given an array of features and target

    Args:
        model (Sequential): The model to be trained.
        xtrain (numpy array): A collection of training data. 
            Assumed shape is (num_samples, None).
        ytrain (numpy array): A collection of one-hot encoded targets.
            Assumed shape is (num_samples, 4)

    Returns:
        avg_loss (float): The average loss from each instance.
        avg_acc (TYPE): The average accuracy from each instance.

    '''
    losses = list()
    accs = list()
    for feature, label in zip(xtrain, ytrain):
        #reshape input data
        timesteps = len(feature)
        feature = np.reshape(feature, (1, timesteps))
        label = np.reshape(label, (1,4))
        #train
        history = model.fit(feature, label, batch_size=1, epochs=1, 
                            verbose=0, class_weight = class_weights)
        #record stats
        losses.append(history.history['loss'])
        accs.append(history.history['accuracy'])
    avg_loss = np.mean(losses)
    avg_acc = np.mean(accs)
    return avg_loss, avg_acc


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
        avg_acc(float): The avergage accuracy accross each prediction.

    """
    losses = list()
    accs = list()
    for feature, label in zip(xtest, ytest):
        #reshape input data
        timesteps = len(feature)
        feature = np.reshape(feature, (1, timesteps))
        label = np.reshape(label, (1,4))
        #test
        loss, acc = model.evaluate(feature, label, verbose=0)
        #record stats
        losses.append(loss)
        accs.append(acc)
    avg_loss = np.mean(losses)
    avg_acc = np.mean(accs)
    return avg_loss, avg_acc


def get_preds(model, X):
    """Get model predictions on an input.

    Args:
        model (Keras model): The model that will be making predictions.
        X (numpy array): The input for the model to predict on. 
            Assumed shape is (num_samples, None)

    Returns:
        preds (list): The class predictions for each instance.
        probs (list): The probability distribution for each instance.

    """
    preds = list()
    probs = list()
    for feature in X:
        timesteps = len(feature)
        feature = np.reshape(feature, (1, timesteps))
        prob = model.predict_on_batch(feature)
        probs.append(prob)
        preds.append(np.argmax(prob))
    return preds, probs
    

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
floattarget = list(ratings.mean(axis=1))
target = [int(round(num)) for num in floattarget]

#tokenize and encode sentences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(interviews_clean)

vocab_size = len(tokenizer.word_index)
print(f'Found {vocab_size} tokens.')

#compute class weights
class_weights = class_weight.compute_class_weight(class_weight='balanced', 
                                                  classes=np.unique(target), 
                                                  y=np.array(target))

#params for network
dimensionality = 30 #dimensionality of word vectors
lstm_units = 20
vocab_size += 1
    
n_splits = 5

#input must be an array for KFold
X = np.array(tokenizer.texts_to_sequences(interviews_clean))
target = np.array(target)

#lists to track performance accross K folds
all_accs = list()
all_losses = list()
all_preds = list()
all_probs = list()
all_trues = list()

for train_idx, test_idx in KFold(n_splits).split(X):
    print('-'*40)
    print('-'*40)
    
    #to work with function as-is, convert back to list
    #TODO: find a way to not do this
    ytrain = list(target[train_idx])
    ytest = list(target[test_idx])
    
    #convert encoded sentences into xtrain and xtest arrays
    xtrain = X[train_idx]
    xtest = X[test_idx]

    #one-hot encode target
    ytrain = np_utils.to_categorical(ytrain, num_classes=4)
    ytest = np_utils.to_categorical(ytest, num_classes=4) 
    
    #create model 
    model = create_model(vocab_size, dimensionality, lstm_units)
    
    #track epochs manually
    epochs = 20
    
    #train the model
    print('Training model now...')
    for e in range(1, (epochs+1)):
        xtrain, ytrain = shuffle(xtrain, ytrain)
        print(f'Epoch {e}/{epochs}:')
        avg_loss, avg_acc = train(model, xtrain, ytrain, class_weights)
        val_loss, val_acc = test(model, xtest, ytest)
        print(f'\tTrain loss: {avg_loss:.4f} Train Accuracy: {avg_acc:.4f}')
        print(f'\tVal loss: {val_loss:.4f} Val Accuracy: {val_acc:.4f}')
        
    #test the model
    print('-'*40)
    test_loss, test_acc = test(model, xtest, ytest)
    print(f'Test loss: {test_loss:.3f}')
    print(f'Test accuracy: {test_acc:.3f}')
    
    all_accs.append(test_acc)
    all_losses.append(test_loss)
    
    #get predictions for confusion matrix and classification report
    preds, probs = get_preds(model, xtest)
    trues = np.argmax(ytest, axis=1)
    
    all_preds.extend(preds)
    all_probs.extend(probs)
    all_trues.extend(trues)
    
    del model
    

#print average loss and accuracy accross all folds
avg_loss = np.mean(all_losses)
loss_stdv = np.std(all_losses)

avg_acc= np.mean(all_accs)
acc_stdv = np.std(all_accs)

print(f'Average {n_splits}-fold CV loss: {avg_loss:.3f} +/-({loss_stdv:.3f})')
print(f'Average {n_splits}-fold CV accuracy: {avg_acc:.3f} +/-({acc_stdv:.3f})')


#create confusion matrix with preds accross all folds
class_labels = [0,1,2,3]
create_confusion_matrix(all_trues, all_preds, class_labels)

print(classification_report(all_trues, all_preds))

#create dataframe of target, predicted, and probabilities
val_results = pd.DataFrame()
val_results['Target'] = all_trues
val_results['Predicted'] = all_preds
val_results['Probabilities'] = all_probs

#save to csv
val_results.to_csv(save_csv_path)