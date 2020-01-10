import numpy as np
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model 
#%%
#This file is found in the 'Models and Tokenizer' folder
pkl_file = open('tokenizer.pickle', 'rb')
tk = pickle.load(pkl_file) 
pkl_file.close()
#%%
#This file is found in the 'Models and Tokenizer' folder
models = []
classes = ['toxic','severe_toxic','obscene','threat','identity_hate']
for i in classes:
    models.append(load_model(i + '.h5'))
#%%
INPUT = 'You are a bitch'
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
if len(labels) != 0:
    print(labels)
else:
    print("Neither offensive nor sensitive")
#print(preds)   
#%%