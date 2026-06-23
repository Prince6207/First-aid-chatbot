import nltk
nltk.download('punkt_tab')
from nltk.stem.lancaster import LancasterStemmer
stemmer=LancasterStemmer()
import json
import random
with open('intents.json') as f:
    data=json.load(f)
print(data)
words=[]
docs_x=[]
docs_y=[]
labels=[]
for intent in data['intents']:
    for pattern in intent['patterns']:
        wrds=nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(pattern)
        docs_y.append(intent['tag'])
    if intent['tag'] not in labels:
        labels.append(intent['tag'])
words=[stemmer(w.lower()) for w in words]
print(words)