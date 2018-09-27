import pandas as pd
import numpy as np
import os
import sys
import xml.etree.ElementTree as ET
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.initializers import Constant
from keras.layers import Dropout
from keras import regularizers
from keras import backend as K
from keras.models import Sequential
from keras.layers.recurrent import GRU
from keras.models import Sequential
from keras.layers.recurrent import GRU
import keras.losses
import tensorflow as tf
import keras.backend.tensorflow_backend as tfb
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt

MAX_NUM_WORDS = 20000 # max. size of vocabulary
EMBEDDING_DIM = 100 # dimension of GloVe word embeddings
MAX_SEQUENCE_LENGTH = 1000 # truncate examples after MAX_SEQUENCE_LENGTH words
CROSS_VALIDATION_SPLIT = 0.1
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
    xmlstring = xmlstring.replace('""', '"')
    text = None
    try: 
        tree = ET.ElementTree(ET.fromstring(xmlstring))
        xpath_result = tree.findall(".//description")
        text = xpath_result[0].text
    except:
        text = xmlstring
    # remove stopwords and punctuation. lower case everything
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    tokens = [w.lower() for w in tokens if not w in stop_words and w.isalpha() and wordnet.synsets(w)]
    # lemmatize
    lemma = WordNetLemmatizer()
    final_tokens = []
    for word in tokens:
        final_tokens.append(lemma.lemmatize(word))
    ret = " ".join(final_tokens)
    return ret
 
data_df["clean_abstract"] = data_df["abstract"].apply(cleanup_abstract)
data_df["clean_concepts"] = data_df["concepts"].apply(lambda x: list({c[c.rfind("/")+1:c.rfind("/")+3] for c in x.split(";") if c[c.rfind("/")+1:c.rfind("/")+3] in EUROVOC_FIELDS}))
data_df = data_df[data_df.astype(str)['clean_concepts'] != '[]']
data_df.drop(["abstract"], axis=1)
data_df.drop(["concepts"], axis=1)
print(data_df[['clean_abstract', 'clean_concepts']][:10])

tmp = []
data_df["clean_concepts"].apply(lambda x: tmp.extend(x))
s = pd.Series(data=tmp)
grouped_labels = s.groupby(s).size().reset_index(name='count')
print(grouped_labels)
    

multilabel_binarizer = MultiLabelBinarizer()
labels = multilabel_binarizer.fit_transform(data_df["clean_concepts"])
tmp_labels = multilabel_binarizer.classes_
for (i, label) in enumerate(multilabel_binarizer.classes_):
    print("{}. {}".format(i + 1, label))

#grouped_labels['class_weight'] = len(grouped_labels) / grouped_labels['count']
grouped_labels['class_weight'] = 1 / grouped_labels['count']
class_weight = {}
for index, label in enumerate(tmp_labels):
    class_weight[index] = grouped_labels[grouped_labels['index'] == label]['class_weight'].values[0]    
print(grouped_labels.head())
print(class_weight)

data = data_df["clean_abstract"].tolist()
tokenizer = Tokenizer(nb_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(data)
sequences = tokenizer.texts_to_sequences(data)
word_index = tokenizer.word_index
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print('Found %s unique tokens.' % len(word_index))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a cross validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(CROSS_VALIDATION_SPLIT * data.shape[0])

train_data = data[:-num_validation_samples]
train_labels = labels[:-num_validation_samples]
cross_data = data[-num_validation_samples:]
cross_labels = labels[-num_validation_samples:]
print(num_validation_samples)
print("train_data.shape", train_data.shape)
print("train_labels.shape", train_labels.shape)
print("cross_data.shape", cross_data.shape)
print("cross_labels.shape", cross_labels.shape)


embeddings_index = {}
with open(os.path.join('glove.6B', 'glove.6B.100d.txt'), 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
print('Found %s word vectors.' % len(embeddings_index))
        
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

def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def f1(y_true, y_pred):
    def prec(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        prec = true_positives / (predicted_positives + K.epsilon())
        return prec

    def rec(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        rec = true_positives / (possible_positives + K.epsilon())
        return rec

    precision = prec(y_true, y_pred)
    recall = rec(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

def hamming(y_true, y_pred):
        denominator = K.sum(K.ones(shape=K.shape(K.flatten(y_true))))
        nominator = K.sum((y_true * (1-K.round(K.clip(y_pred, 0, 1))) + (1-y_true) * K.round(K.clip(y_pred, 0, 1))))
        return (nominator / denominator)

VALIDATION_SPLIT = 0.2 # ration for split of training data and test data
NUM_EPOCHS = 200 # number of epochs the network is trained
DROPOUT = 0.2
#REGULARIZATION = 0.1
BATCH_SIZE = 64
LR = 0.005

model = Sequential()
model.add(Embedding(MAX_NUM_WORDS, EMBEDDING_DIM,input_length = MAX_SEQUENCE_LENGTH))
model.add(GRU(128, dropout=0.25, return_sequences=True))
model.add(GRU(128, dropout=0.25))
model.add(Dense(labels.shape[1], activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[hamming, f1, precision, recall])

history = model.fit(train_data, train_labels, class_weight=class_weight, validation_split=VALIDATION_SPLIT,
          epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)

model.save("models/model_EP_%s_DO_%s_BAT_%s_LR_%s.h5" % (str(NUM_EPOCHS), 
                                                        str(DROPOUT), 
                                                        str(BATCH_SIZE), 
                                                        str(LR)))

loss = history.history['loss']
val_loss = history.history['val_loss']
ham = history.history['hamming']
val_ham = history.history['val_hamming']
f1 = history.history['f1']
val_f1 = history.history['val_f1']
prec = history.history['precision']
val_prec = history.history['val_precision']
rec = history.history['recall']
val_rec = history.history['val_recall']

epochs = range(len(ham))                
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=1, ncols=5, figsize=(25, 5))
ax1.plot(epochs, ham, label='Training ham')
ax1.plot(epochs, val_ham, label='Validation ham')
ax1.set_ylabel('ham')
ax1.set_xlabel('epoch')
ax1.set_title('Hamming')
ax1.legend()
ax2.plot(epochs, loss, label='Training loss')
ax2.plot(epochs, val_loss, label='Validation loss')
ax2.set_ylabel('loss')
ax2.set_xlabel('epoch')
ax2.set_title('Loss')
ax2.legend()
ax3.plot(epochs, f1, label='Training f1')
ax3.plot(epochs, val_f1, label='Validation f1')
ax3.set_ylabel('f1')
ax3.set_xlabel('epoch')
ax3.set_title('F1')
ax3.legend()

ax4.plot(epochs, prec, label='Training precision')
ax4.plot(epochs, val_prec, label='Validation precision')
ax4.set_ylabel('precision')
ax4.set_xlabel('epoch')
ax4.set_title('Precision')
ax4.legend()

ax5.plot(epochs, rec, label='Training recall')
ax5.plot(epochs, val_rec, label='Validation recall')
ax5.set_ylabel('recall')
ax5.set_xlabel('epoch')
ax5.set_title('Recall')
ax5.legend()
fig.savefig("figures/model_EP_%s_DO_%s_BAT_%s_LR_%s.png" % (str(NUM_EPOCHS), 
                                                        str(DROPOUT), 
                                                        str(BATCH_SIZE), 
                                                        str(LR)))

score = model.evaluate(cross_data, cross_labels, batch_size=BATCH_SIZE)
print('Cross loss:', score[0])
print('Cross hamming:', score[1])
print('Cross f1:', score[2])
print('Cross precision:', score[3])
print('Cross recall:', score[4])
