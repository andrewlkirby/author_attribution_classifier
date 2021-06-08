import re
import pandas as pd
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import keras
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

tpc_path = 'the_perfect_crime.txt'
jul_path = 'julian.txt'
obv_path = 'oblivion.txt'

seed = 125

def tpc_clean(path):
    with open(path, 'r') as fin, open('tpc_clean.txt', 'w') as fout:
        for i in fin:
            tpc_txt = re.sub('\n*\-{2}\s\d+\s\-{2}\n*', '\n', i)
            tpc_txt = re.sub('\n*\-{2}\s\[\d*\]\s\-{2}\n*', '\n', tpc_txt)
            tpc_txt = re.sub('\n\s\n\-{2}.*\n.*', ' ', tpc_txt)
            tpc_txt = re.sub('\n*\-{2}\s[a-zA-z]*\s\-{2}\s\n*', ' ', tpc_txt)
            fout.write(tpc_txt)            

def jul_clean(path):
    with open(path, 'r') as fin, open('jul_clean.txt', 'w') as fout:
        for i in fin:
            jul_txt = re.sub('•', '', i)
            fout.write(jul_txt)
            
def __main__():
    tpc_clean(tpc_path)
    jul_clean(jul_path)

__main__()

#Sentence Boundary Detection w/DetectorMorse:
#!python -m detectormorse -V -r -s=in.txt > out.txt
#Output files: tpc_sbd.txt, jul_sbd.txt, obv_sbd.txt


#open file and send to dataframe; assign class 1, 2, 3:
#https://stackoverflow.com/questions/3277503/how-to-read-a-file-line-by-line-into-a-list

with open(r'tpc_sbd.txt') as f:
    content = f.readlines()
    content = [x.strip() for x in content] 
    df_tpc = pd.DataFrame(content)
    df_tpc['class'] = 1
    #print(df_tpc)
    
with open(r'jul_sbd.txt') as f:
    content = f.readlines()
    content = [x.strip() for x in content] 
    df_jul = pd.DataFrame(content)
    df_jul['class'] = 2
    #print(df_jul)
    
with open(r'obv_sbd.txt') as f:
    content = f.readlines()
    content = [x.strip() for x in content] 
    df_obv = pd.DataFrame(content)
    df_obv['class'] = 3
    #print(df_obv)
    
print("\n")
print("Sample from The Perfect Crime: ", df_tpc.iloc[2000, 0])
print("\n")
print("Sample from Julian: ", df_jul.iloc[2000, 0])
print("\n")
print("Sample from Oblivion: ", df_obv.iloc[2000, 0])
print("\n")

#shuffle dataframes and take first 3000 sentences from each:
df_tpc = shuffle(df_tpc, random_state=seed)
df_jul = shuffle(df_jul, random_state=seed)
df_obv = shuffle(df_obv, random_state=seed)

df_tpc = df_tpc[0:3000]
df_jul = df_jul[0:3000]
df_obv = df_obv[0:3000]

df = pd.concat([df_tpc, df_jul, df_obv], axis = 0)
df = df.rename(columns={0: 'sents'})
#print(df)

X = df['sents']
y = df['class']

#TF-IDF
count_vectorizer = CountVectorizer(ngram_range = (1, 1))
tfidf = TfidfTransformer()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=seed)

count_matrix_train = count_vectorizer.fit_transform(X_train)
X_train_tfidf = tfidf.fit_transform(count_matrix_train)
X_train_tfidf = X_train_tfidf.todense()

count_matrix_test = count_vectorizer.transform(X_test)
X_test_tfidf = tfidf.transform(count_matrix_test)
X_test_tfidf = X_test_tfidf.todense()

clf = LogisticRegression(penalty='l2', n_jobs=1, multi_class='multinomial', 
                         C=1.0, solver ='newton-cg'
                         ).fit(X_train_tfidf, y_train)

print("\n")
print("score with TF-IDF: ", clf.score(X_test_tfidf, y_test))

# KERAS + GLOVE
#https://keras.io/examples/nlp/pretrained_word_embeddings/
df = shuffle(df, random_state=seed)

samples = df['sents'].to_list()
labels = df['class'].to_list()

validation_split = 0.2
num_validation_samples = int(validation_split * len(samples))
train_samples = samples[:-num_validation_samples]
val_samples = samples[-num_validation_samples:]
train_labels = labels[:-num_validation_samples]
val_labels = labels[-num_validation_samples:]


#vect text
vectorizer = TextVectorization(max_tokens=20000, output_sequence_length=200)
text_ds = tf.data.Dataset.from_tensor_slices(train_samples).batch(128)
vectorizer.adapt(text_ds)

#download glove: https://nlp.stanford.edu/projects/glove/
path_to_glove_file = 'glove.6B.100d.txt'

embeddings_index = {}
with open(path_to_glove_file) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

voc = vectorizer.get_vocabulary()
word_index = dict(zip(voc, range(len(voc))))

num_tokens = len(voc) + 2
embedding_dim = 100
hits = 0
misses = 0

# Prepare embedding matrix
embedding_matrix = np.zeros((num_tokens, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        # This includes the representation for "padding" and "OOV"
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1

embedding_layer = Embedding(
    num_tokens,
    embedding_dim,
    embeddings_initializer=keras.initializers.Constant(embedding_matrix),
    trainable=False
    )

#prepare layers 
int_sequences_input = keras.Input(shape=(None,), dtype="int64")
embedded_sequences = embedding_layer(int_sequences_input)
x = layers.Conv1D(128, 5, activation="relu")(embedded_sequences)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(128, 5, activation="relu")(x)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(128, 5, activation="relu")(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)
preds = layers.Dense(len(df), activation="softmax")(x)
model = keras.Model(int_sequences_input, preds)
#model.summary()

x_train = vectorizer(np.array([[s] for s in train_samples])).numpy()
x_val = vectorizer(np.array([[s] for s in val_samples])).numpy()

y_train = np.array(train_labels)
y_val = np.array(val_labels)

model.compile(
    loss="sparse_categorical_crossentropy", optimizer="rmsprop", metrics=["acc"]
)
model.fit(x_train, y_train, batch_size=128, epochs=25, validation_data=(x_val, y_val))

#beta model

val = 9000

y1 = [1] * int((val/3))
y2 = [2] * int((val/3))
y3 = [3] * int((val/3))

y = y1 + y2 + y3

np.random.seed(seed)

x1 = (np.random.randint(1, 6, size = (val, 1)))
x2 = (np.random.randint(1, 6, size = (val, 1)))
x3 = (np.random.randint(1, 6, size = (val, 1)))
x4 = (np.random.randint(1, 6, size = (val, 1)))
x5 = (np.random.randint(1, 6, size = (val, 1)))

X = np.c_[x1, x2, x3, x4, x5]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=seed)

clf = LogisticRegression(penalty='l2', n_jobs=1, multi_class='multinomial', 
                         C=1.0, solver ='newton-cg').fit(X_train, y_train)

print("\n")
print("beta model score: ", clf.score(X_test, y_test))
