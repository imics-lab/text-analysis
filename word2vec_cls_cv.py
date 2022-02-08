# -*- coding: utf-8 -*-
"""Performs 5 fold cross validation on a classifier with a word2vec 
embedding layer.

Produces a confusion matrix and classification report of predictions 
aggregated accross all folds.

Created on Fri Jul 10 14:56:30 2020

@author: Morgan Byers
"""

from configparser import ConfigParser
from gensim.models import Word2Vec
from keras.layers import Dense, Embedding, LSTM, Dropout
from keras.models import Sequential
from keras.utils import np_utils
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.utils import shuffle, class_weight
import tensorflow as tf
import ordinal_categorical_crossentropy as OCC


def preprocess_data(interviews):
    '''Cleans and tokenizes data.
    
    Cleans data by removing all punctuation and numbers. Turns all letters 
    to lowercase.

    Args:
        interviews (list): The corpus to be cleaned.

    Returns:
        interviews (list): The cleaned and tokenized corpus.

    '''
    tokens = [word_tokenize(sent) for sent in interviews]
    
    clean_tokens = list()
    for sentence in tokens:
        clean_sentence = list()
        for word in sentence:
            if word.isalpha():
                clean_sentence.append(word)
        clean_tokens.append(clean_sentence)
    
    lower_tokens = [[word.lower() for word in sent] for sent in clean_tokens]
    
    return lower_tokens


def encode_sentences(encoder, sentences):
    """Encode sentences using the encoder provided.
    
    Words that are not in the encoder's vocabulary are encoded with a 
    special 'unk' token.

    Args:
        encoder (models.word2vec.Word2Vec): The trained encoder.
        sentences (list): A list of tokenized sentences to be encoded.

    Returns:
        encoded_sentences (list): A list of the now encoded sentences.

    """
    encoded_sentences = list()  
    
    for i, sentence in enumerate(sentences):
        encoded_sentence = list()
        for word in sentence:
            try:
                encoding = encoder.wv.vocab[word].index
                encoded_sentence.append(encoding)
            except:
                #encode OOV words with unk token
                encoding = encoder.wv.vocab['unk'].index
                encoded_sentence.append(encoding)
        if encoded_sentence:
            encoded_sentences.append(encoded_sentence)
        else:
            #one response consists of only '7'. This line 
            # specifically addresses that sentence. 
            encoded_sentences.append([encoder.wv.vocab['unk'].index])
    return encoded_sentences


def train_embedding(train_tokens, dim, min_ct, num_threads, num_iter=50):
    """Train the embedding layer on tokenized corpus provided.

    Args:
        train_tokens (list): Tokenized colection of sentences.
        dim (int): The dimensionality of the word vectors.
        min_ct (int): The minimum number of times a word must appear 
            in the corpus to be included in the final encoding 
            of the sentences.
        num_threads (int): The number of threads to use when 
            training embedding layer.
        num_iter (int, optional): The number of iterations used to train 
            the embedding layer. Defaults to 50.

    Returns:
        encoder (TYPE): The trained embedding layer.

    """
    encoder = Word2Vec(sentences=train_tokens, size=dim, min_count=min_ct, 
                       workers=num_threads, iter=num_iter)
    return encoder
    

def create_model(pretrained_weights, vocab_size, dimensionality, lstm_units):
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
    model.add(Embedding(vocab_size, dimensionality, 
                        weights=[pretrained_weights], trainable=False))
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
        class_weights (TYPE): Class weights for training model on skewed data.

    Returns:
        avg_loss (float): The average loss from each instance.
        avg_acc (float): The average accuracy from each instance.

    '''
    losses = list()
    accs = list()
    for feature, label in zip(xtrain, ytrain):
        #reshape input data
        print(feature)
        timesteps = len(feature)
        feature = np.reshape(feature, (1, timesteps))
        print(feature)
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
        avg_acc (float): The average accuracy accross each prediction.

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
    '''Returns predicted class and all class probabilities of a 
        model's predictions on X.

    Args:
        model (Keras Model): The model that will be making the predictions.
        X (numpy array): A collection of input data. 
            Assumed shape is (num_samples, none)

    Returns:
        preds (list): The model's prediction(s), rounded to the integer class.
        probs (list): The probability distribution(s) of the model's 
            prediction(s).

    '''
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

#compute class weights
class_weights = class_weight.compute_class_weight(class_weight='balanced', 
                                                  classes=np.unique(target), 
                                                  y=np.array(target))

#set some parameters for encoder
dimensionality = 30 #dimensionality of word vectors
num_threads = 6 #number of threads to use
min_ct = 1

#params for network
lstm_units = 20
    
n_splits = 5

#input must be an array for KFold
word_tokens = np.array(interviews_clean)
target = np.array(target)

#lists to track performance accross K folds
all_accs = list()
all_losses = list()
all_preds = list()
all_probs = list()
all_trues = list()

for train_idx, test_idx in KFold(n_splits).split(word_tokens):
    print('-'*40)
    print('-'*40)
    #to work with functions as-is, convert back to list
    #TODO: find a way to not do this
    train_tokens = list(word_tokens[train_idx])
    test_tokens = list(word_tokens[test_idx])
    ytrain = list(target[train_idx])
    ytest = list(target[test_idx])
    
    #create list of training tokens with an extra OOV token
    training_set = train_tokens.copy()
    
    #TODO: if min count is raised, unk token needs to be 
    # increased accordingly. FIX THIS
    training_set.append(['unk'])
    
    #train embedding layer
    encoder = train_embedding(training_set, dimensionality, 
                              min_ct, num_threads)
    
    #get vocab length
    vocab_size = len(encoder.wv.vocab)
    print(f'Found {vocab_size} unique tokens')

    #encode the sentences in 2 batches - train and test
    train_encoded_sentences = encode_sentences(encoder, train_tokens)
    test_encoded_sentences = encode_sentences(encoder, test_tokens)
    
    #convert encoded sentences into xtrain and xtest arrays
    xtrain = np.array(train_encoded_sentences)
    xtest = np.array(test_encoded_sentences)
        
    #get pretrained weights for embedding layer
    pretrained_weights = encoder.wv.vectors

    #one-hot encode target
    ytrain = np_utils.to_categorical(ytrain, num_classes=4)
    ytest = np_utils.to_categorical(ytest, num_classes=4) 
    
    #create model and show summary
    model = create_model(pretrained_weights, vocab_size, dimensionality, 
                         lstm_units)
    
    #track epochs manually
    epochs = 20
    
    #train the model
    print('Training model now...')
    for e in range(1, (epochs+1)):
        xtrain, ytrain = shuffle(xtrain, ytrain)
        print(f'Epoch {e}/{epochs}:')
        avg_loss, avg_acc = train(model, xtrain, ytrain, class_weights)
        val_loss, val_acc = test(model, xtest, ytest)
        #print(f'\tTrain loss: {avg_loss:.4f} Train Accuracy: {avg_acc:.4f}')
        #print(f'\tVal loss: {val_loss:.4f} Val Accuracy: {val_acc:.4f}')
        
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