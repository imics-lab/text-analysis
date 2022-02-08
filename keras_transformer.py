# -*- coding: utf-8 -*-
"""Performs text classification using a transformer model.

Created on Wed Sep  2 19:13:07 2020

@author: Morgan Byers
"""

from configparser import ConfigParser
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import re
import string
from sklearn.model_selection import KFold
from sklearn.utils import class_weight, shuffle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
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
        plt.title(f'{metric} over training epochs')
    plt.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), 
               shadow=True)
    plt.show()


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
    

class MultiHeadSelfAttention(layers.Layer):
    
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            error_msg = f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            raise ValueError(error_msg)
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output
    

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, 
                                          output_dim=embed_dim, mask_zero=True)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


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

#pad input
X = pad_sequences(encoded_sentences, maxlen=max_len)

#target becomes categorical
y = np_utils.to_categorical(target, num_classes=4)

#create model
embed_dim = 100  # Embedding size for each token
num_heads = 5  # Number of attention heads
ff_dim = 64  # Hidden layer size in feed forward network inside transformer
vocab_size += 1

n_splits = 5

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

    inputs = layers.Input(shape=(max_len))
    embedding_layer = TokenAndPositionEmbedding(max_len, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(25, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(4, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer='adam', loss=OCC.loss, metrics=['accuracy'])
    
    class_weights = class_weight.compute_class_weight(class_weight='balanced', 
                                                      classes=np.unique(target), 
                                                      y=np.array(target))
    
    weight_dict = {
        0: class_weights[0],
        1: class_weights[1],
        2: class_weights[2],
        3: class_weights[3]
        }

    history = model.fit(xtrain, ytrain, batch_size=32, epochs=10, verbose=0, 
                        validation_split=0.2, class_weight=weight_dict)
    
    histories.append(history)

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
