import pandas as pd  
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.layers import LSTM
import re
from sklearn.metrics import precision_recall_fscore_support
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
#%%
train = pd.read_csv('train.csv')
train['comment_text'] = train['comment_text'].map(lambda x: re.sub('\\n',' ',str(x)))
train['comment_text'] = train['comment_text'].map(lambda x: re.sub("\[\[User.*",'',str(x)))
train['comment_text'] = train['comment_text'].map(lambda x: re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",'',str(x)))
train['comment_text'] = train['comment_text'].map(lambda x: re.sub("(http://.*?\s)|(http://.*)",'',str(x)))
#%%
def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text
train['comment_text'] = train['comment_text'].map(lambda com : clean_text(com))
x = train['comment_text'].values
y = train.iloc[:, 2:8].values 
#%%
X = []
for i in x:
    X.append([w for w in i.split() if w not in stop_words])
#%%
X = np.array(X) 
print(X.shape)
y = np.array(y)
print(y.shape)
#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=13)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
#%%
tk = Tokenizer(num_words=30000,
filters='!"#$%&()*+,-./:;<=>?@[\]^_`{"}~\t\n',lower=True, split=" ")
tk.fit_on_texts(X_train)
X_train_seq = tk.texts_to_sequences(X_train)
X_test_seq = tk.texts_to_sequences(X_test)
#%%
lengths = []
for i in X_train_seq:
    lengths.append(len(i))
pd.Series(lengths).describe()
#%%
X_train_seq_trunc = pad_sequences(X_train_seq, maxlen=36)
X_test_seq_trunc = pad_sequences(X_test_seq, maxlen=36)
#%%
models = []
classes = ['toxic','severe_toxic','obscene','threat','identity_hate']
#%%
for i in range(len(classes)):
    model = Sequential()
    model.add(Embedding(30000, 18, input_length=36))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train_seq_trunc, y_train[:,i], epochs=2, batch_size=512)
    _, accuracy = model.evaluate(X_test_seq_trunc, y_test[:,i])
    print('For ' + classes[i])
    print(accuracy)
    res = model.predict_classes(X_test_seq_trunc)
    print(precision_recall_fscore_support(y_test[:,i], res, average='macro'))
    models.append(model)
#%%
INPUT = 'You are full of shit bitch'
t = tk.texts_to_sequences([INPUT])[0]
if len(t) <= 4:
    test = t+t+t
elif len(t) <= 8:
    test = t+t
else:
    test = t
pad = pad_sequences([test], maxlen=36, padding='post')
labels = []
preds = []
for i in range(len(classes)):
    yhat = models[i].predict_classes(np.array(pad))
    preds.append(float(models[i].predict(np.array(pad))[0]))
    if(yhat[0] == 1):
        labels.append(classes[i])
print(labels)
print(preds)        
#%%
for i in range(len(models)):
    models[i].save(classes[i] + '.h5')    
#%%