import pandas as pd
import numpy as np
import nltk
import re
import warnings
import gensim
from matplotlib import pyplot
from sklearn.decomposition import PCA
from gensim.models import Word2Vec,KeyedVectors
from gensim.test.utils import lee_corpus_list
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

warnings.filterwarnings("ignore")

data=pd.read_csv('spam_or_not_spam.csv')
tdata = data
docs=list()
for i in range(len(tdata)):
    email=[]
    email=tdata['email'][i].split()
    for word in email:
        if(not re.search('[a-zA-Z]', word)):
            email.remove(word)
    docs.append(' '.join(email))

labels=tdata['label'].values

t = Tokenizer() 
t.fit_on_texts(docs)
vocab_size = len(t.word_index) + 1
encoded_docs = t.texts_to_sequences(docs)
max_len=0
for seq in encoded_docs:
    word_num=len(seq)
    if(word_num>max_len):
        max_len=word_num
print(max_len)

padocs=pad_sequences(encoded_docs,maxlen=max_len,padding='pre')
#print(docs)

embeddings_index = dict()
f = open('glove.6B.100d.txt',encoding='utf-8') 
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print(embeddings_index['there'])
#print(t.word_index.items())

embed_matrix=np.zeros((vocab_size,100))
for word, vec in t.word_index.items():
    embed_vector = embeddings_index.get(word)
    if embed_vector is not None:
        embed_matrix[vec] = embed_vector

print(embed_matrix.shape)

model = Sequential()
model.add(Embedding(vocab_size, 100, weights=[embed_matrix], input_length=max_len, trainable=False))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
# summarize the model
print(model.summary())

X_train, X_test, y_train, y_test = train_test_split(padocs, labels, test_size=0.25, random_state=42)
print(X_test)
model.fit(X_train, y_train, epochs=15, verbose=0)
ypred=model.predict_classes(X_test)
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy: %f' % (accuracy*100))
print(len(y_test))
print(len(ypred))
print(classification_report(y_test, ypred))

y_pred=list()
for i in range(len(ypred)):
    for j in range(len(ypred[i])):
        y_pred.append(ypred[i][j])

df = pd.DataFrame({'Actual':y_test,'Predicted': y_pred})
print(df)