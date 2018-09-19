import pandas as pd
import numpy as np
import os
import sys
import xml.etree.ElementTree as ET
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.initializers import Constant
from keras.layers import Dropout
from keras import regularizers
import keras.losses

import tensorflow as tf
import keras.backend.tensorflow_backend as tfb

from sklearn.preprocessing import MultiLabelBinarizer

import matplotlib.pyplot as plt

CLEANUP_DATA = True # save time by loading a cleaned up version from disc
MAX_NUM_WORDS = 20000 # max. size of vocabulary
EMBEDDING_DIM = 100 # dimension of GloVe word embeddings
MAX_SEQUENCE_LENGTH = 500 # truncate examples after MAX_SEQUENCE_LENGTH words
VALIDATION_SPLIT = 0.2 # ration for split of training data and test data
NUM_EPOCHS = 10 # number of epochs the network is trained
# those are the eurovoc fields we want to support
EUROVOC_FIELDS = {
"4":  "POLITICS",
"8":  "INTERNATIONAL RELATIONS", 
"10": "EUROPEAN UNION",
"12": "LAW ",
"16": "ECONOMICS",
"20": "TRADE",
"24": "FINANCE",
"28": "SOCIAL QUESTIONS",
"32": "EDUCATION AND COMMUNICATIONS",
"36": "SCIENCE",
"40": "BUSINESS AND COMPETITION",
"44": "EMPLOYMENT AND WORKING CONDITIONS",
"48": "TRANSPORT",
"52": "ENVIRONMENT",
"56": "AGRICULTURE, FORESTRY AND FISHERIES",
"60": "AGRI-FOODSTUFFS",
"64": "PRODUCTION, TECHNOLOGY AND RESEARCH",
"66": "ENERGY",
"68": "INDUSTRY",
"72": "GEOGRAPHY",
"76": "INTERNATIONAL ORGANISATIONS"
}

data_df = pd.read_csv("data.csv")

def cleanup_abstract(xmlstring):
    #import ipdb; ipdb.set_trace()
    xmlstring = xmlstring.replace('""', '"')
    text = None
    try: 
        tree = ET.ElementTree(ET.fromstring(xmlstring))
        xpath_result = tree.findall(".//description")
        text = xpath_result[0].text
    except:
        text = xmlstring
    return text


if CLEANUP_DATA:
    data_df["clean_abstract"] = data_df["abstract"].apply(cleanup_abstract)
    data_df["clean_concepts"] = data_df["concepts"].apply(lambda x: list({c[c.rfind("/")+1:c.rfind("/")+3] for c in x.split(";") if c[c.rfind("/")+1:c.rfind("/")+3] in EUROVOC_FIELDS}))
    data_df = data_df[data_df.astype(str)['clean_concepts'] != '[]']
    data_df.drop(["abstract"], axis=1)
    data_df.drop(["concepts"], axis=1)
    

labels = data_df["clean_concepts"].tolist()
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)

data = data_df["clean_abstract"].tolist()
tokenizer = Tokenizer(nb_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(data)
sequences = tokenizer.texts_to_sequences(data)
word_index = tokenizer.word_index
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

embeddings_index = {}
with open(os.path.join('glove.6B', 'glove.6B.100d.txt'), 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
        
num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
    
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

for DROPOUT in [0.1, 0.25]:
    for REGULARIZATION in [1.0, 5.0, 10.0]:
        for BATCH_SIZE in [64]:
            
            params ={ "dropout": str(DROPOUT), 
                      "regularization": str(REGULARIZATION), 
                      "batch_size": str(BATCH_SIZE)}

            sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
            embedded_sequences = embedding_layer(sequence_input)
            x = Conv1D(128, 5, activation='relu', kernel_regularizer=regularizers.l2(REGULARIZATION))(embedded_sequences)
            x = MaxPooling1D(5)(x)
            x = Conv1D(128, 5, activation='relu', kernel_regularizer=regularizers.l2(REGULARIZATION))(x)
            x = MaxPooling1D(5)(x)
            x = Conv1D(128, 3, activation='relu', kernel_regularizer=regularizers.l2(REGULARIZATION))(x)
            #x = MaxPooling1D(5)(x)  # global max pooling
            x = Flatten()(x)
            x = Dropout(DROPOUT)(x)
            x = Dense(128, activation='relu',kernel_regularizer=regularizers.l2(REGULARIZATION))(x)
            x = Dropout(DROPOUT)(x)
            preds = Dense(labels.shape[1], activation='sigmoid', kernel_regularizer=regularizers.l2(REGULARIZATION))(x)

            model = Model(sequence_input, preds)
            model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['categorical_accuracy'])

            print(model.summary())

            history = model.fit(data, labels, validation_split=0.2,
                      epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)          
            
            acc = history.history['categorical_accuracy']
            val_acc = history.history['val_categorical_accuracy']
            loss = history.history['loss']
            val_loss = history.history['val_loss']
            epochs = range(len(acc))
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
            ax1.plot(epochs, acc, label='Training acc')
            ax1.plot(epochs, val_acc, label='Validation acc')
            ax1.set_ylabel('accuracy')
            ax1.set_xlabel('epoch')
            ax1.set_title('Accuracy: PW %(pos_weight)s, DO %(dropout)s, REG %(regularization)s, BATCH %(batch_size)s'% params)
            ax1.legend()
            ax2.plot(epochs, loss, label='Training loss')
            ax2.plot(epochs, val_loss, label='Validation loss')
            ax2.set_ylabel('loss')
            ax2.set_xlabel('epoch')
            ax2.set_title('Loss: PW %(pos_weight)s, DO %(dropout)s, REG %(regularization)s, BATCH %(batch_size)s'%params)
            ax2.legend()
            fig.savefig("figures/model_%(pos_weight)s_%(dropout)s_%(regularization)s_%(batch_size)s.png"%params)
            