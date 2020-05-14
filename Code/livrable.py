#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 16:09:41 2020

@author: anassmellouki
"""
from __future__ import division, print_function
import spacy
from nltk.corpus import stopwords 
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nlp = spacy.load('fr_core_news_md')
import sys
import time



from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json
import numpy as np
import pandas as pd
import string

from nltk import word_tokenize

import pickle
start_time = time.time()
file=str(sys.argv[1])
data = pd.read_excel(file,header=None, sheet_name="Verbatims")


data=data.rename(columns={0: "Text_Clean"})
data=data.dropna()
#data=data.iloc[22600:,:]
data['Text_Clean'] = data['Text_Clean'].str.replace('<LF>','')
data['Text_Clean'] = data['Text_Clean'].str.replace('<SEP>','')
data['Text_Clean'] = data['Text_Clean'].str.replace('<QT>','')
print(data)

data["Text_Clean"] = data['Text_Clean'].str.replace('[^\w\s]','')
data["Text_Clean"]=data['Text_Clean'].str.lower()
print(data)



temp=data["Text_Clean"]
phrase=[]
essai=[]

tempf=[]
temp2=[]

  
stop_words = set(stopwords.words('french')) 
i=0
print("Extraction des stop-words")
for sentence in temp:
    sys.stdout.write('\r')
    # the exact output you're looking for:
    sys.stdout.write("[%-20s] %d%%" % ('='*int(((i*100)/len(temp))/5), (i*100)/len(temp)))
    sys.stdout.flush()
    i=i+1
    mot=[]
    doc=nlp(sentence)
    for token in doc:
        mot.append(token.text)
    mot=" ".join(str(x) for x in mot)
    word_tokens = word_tokenize(mot) 

    for w in word_tokens: 
        if w not in stop_words:
            if w not in string.punctuation:
                temp2.append(w) 
  
    tempf.append(' '.join(temp2))
    temp2=[]


df = pd.DataFrame(tempf)  
data= df.rename(columns={0: "Text_Clean"})



temp=data["Text_Clean"]
i=0

print("Lemmatisation")
for sentence in temp:

    
    sys.stdout.write('\r')
    # the exact output you're looking for:
    sys.stdout.write("[%-20s] %d%%" % ('='*int(((i*100)/len(temp))/5), (i*100)/len(temp)))
    sys.stdout.flush()
    i=i+1
    
    misspelled=sentence.split()
    doc=nlp(sentence)
    for token in doc:
        phrase.append(token.lemma_)
       
    essai.append(' '.join(phrase))
    phrase=[]
    #for word in misspelled:
        # Get the one `most likely` answer
        #if(spell[word]==0):
            #word=spell.correction(word)
            

df = pd.DataFrame(essai)  
data= df.rename(columns={0: "Text_Clean"})

print(data)


data.insert(loc=1, column='Label', value=0)



tokens = [word_tokenize(sen) for sen in data.Text_Clean]


stoplist = stopwords.words('french')
def removeStopWords(tokens): 
    return [word for word in tokens if word not in stoplist]


filtered_words = [removeStopWords(sen) for sen in tokens]
data['Text_Final'] = [' '.join(sen) for sen in filtered_words]
data['tokens'] = filtered_words

pos = []
neg = []
for l in data.Label:
    if l == 0:
        pos.append(0)
        neg.append(1)
    elif l == 1:
        pos.append(1)
        neg.append(0)
data['Pos']= pos
data['Neg']= neg

data = data[['Text_Final', 'tokens', 'Label', 'Pos', 'Neg']]



all_test_words = [word for tokens in data["tokens"] for word in tokens]
test_sentence_lengths = [len(tokens) for tokens in data["tokens"]]
TEST_VOCAB = sorted(list(set(all_test_words)))
print("%s Mots au total, dont %s differents " % (len(all_test_words), len(TEST_VOCAB)))
print("La phrase la plus longue a %s mots" % max(test_sentence_lengths))
MAX_SEQUENCE_LENGTH = 200

with open('Modeles/tokenizer.pickle', 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)
    
test_sequences = loaded_tokenizer.texts_to_sequences(data["Text_Final"].tolist())
test_cnn_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)



# load json and create model
json_file= open('Modeles/organisation.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("Weights/organisation.h5")


print("Prédiction des verbatims -- Organisation")


predictions = loaded_model.predict(test_cnn_data, batch_size=1024, verbose=1)
orga=pd.DataFrame(predictions)[0]
labels = [1, 0]


prediction_labels=[]
for p in predictions:
    prediction_labels.append(labels[np.argmax(p)])


data.insert(loc=5, column='Organisation', value=prediction_labels)


# load json and create model
json_file = open('Modeles/environnement.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("Weights/environnement.h5")


print("Prédiction des verbatims -- Environnement")


predictions = loaded_model.predict(test_cnn_data, batch_size=1024, verbose=1)
envi=pd.DataFrame(predictions)[0]
labels = [1, 0]


prediction_labels=[]
for p in predictions:
    prediction_labels.append(labels[np.argmax(p)])


data.insert(loc=6, column='Environnement', value=prediction_labels)



# load json and create model
json_file = open('Modeles/communication.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("Weights/communication.h5")


print("Prédiction des verbatims -- Communication")


predictions = loaded_model.predict(test_cnn_data, batch_size=1024, verbose=1)
comm=pd.DataFrame(predictions)[0]
labels = [1, 0]


prediction_labels=[]
for p in predictions:
    prediction_labels.append(labels[np.argmax(p)])


data.insert(loc=7, column='Communication', value=prediction_labels)





# load json and create model
json_file = open('Modeles/technique.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("Weights/technique.h5")


print("Prédiction des verbatims -- Technique")


predictions = loaded_model.predict(test_cnn_data, batch_size=1024, verbose=1)
tech=pd.DataFrame(predictions)[0]
labels = [1, 0]


prediction_labels=[]
for p in predictions:
    prediction_labels.append(labels[np.argmax(p)])


data.insert(loc=8, column='Technique', value=prediction_labels)
data=data.iloc[:,[0,5,6,7,8]]
checkpoint=data

po=0
for i in range(0,len(data)):
    if data["Organisation"][i]== 0 and data["Environnement"][i]==0 and data["Communication"][i]==0 and data["Technique"][i]==0:
        po=po+1
        print(i)
        
        
        flag=max(orga[i],envi[i],comm[i],tech[i])
        
        
        if max(orga[i],envi[i],comm[i],tech[i])==orga[i]:
            data["Organisation"][i]=1
        if max(orga[i],envi[i],comm[i],tech[i])==envi[i]:
            data["Environnement"][i]=1
        if max(orga[i],envi[i],comm[i],tech[i])==comm[i]:
            data["Communication"][i]=1
        if max(orga[i],envi[i],comm[i],tech[i])==tech[i]:
            data["Technique"][i]=1
        
nb_zero=0
nb_one=0
nb_two=0
nb_three=0
nb_four=0
for i in range(0,len(data)):
    if data["Organisation"][i]+data["Environnement"][i]+data["Communication"][i]+data["Technique"][i]==0:
        nb_zero=nb_zero+1
    if data["Organisation"][i]+data["Environnement"][i]+data["Communication"][i]+data["Technique"][i]==1:
        nb_one=nb_one+1
    if data["Organisation"][i]+data["Environnement"][i]+data["Communication"][i]+data["Technique"][i]==2:
        nb_two=nb_two+1
    if data["Organisation"][i]+data["Environnement"][i]+data["Communication"][i]+data["Technique"][i]==3:
        nb_three=nb_three+1  
    if data["Organisation"][i]+data["Environnement"][i]+data["Communication"][i]+data["Technique"][i]==4:
        nb_four=nb_four+1     
        
text = pd.read_excel(file,header=None, sheet_name="Verbatims")
text=text.rename(columns={0: "Texte"})
text=text.dropna()        

data["Text_Final"]=text["Texte"] 

      
data.to_csv("resultat.csv", sep=';',index=False)

nb_verbatim=len(data)
nb_orga=sum(data["Organisation"])
nb_envi=sum(data["Environnement"])
nb_comm=sum(data["Communication"])
nb_tech=sum(data["Technique"])


print("%s Verbatims ont été classé au total" % nb_verbatim)
print("%s Verbatims ont été classé dans la classe ORGANISATION" % nb_orga)
print("%s Verbatims ont été classé dans la classe ENVIRONNEMENT" % nb_envi)
print("%s Verbatims ont été classé dans la classe COMMUNICATION" % nb_comm)
print("%s Verbatims ont été classé dans la classe TECHNIQUE" % nb_tech)

print("")
print ("------------------------------------------------")
print ("------------------------------------------------")
print("")

print("%s Verbatims ont été classé dans aucune classe" % nb_zero)
print("%s Verbatims ont été classé dans 1 classe" % nb_one)
print("%s Verbatims ont été classé dans 2 classes" % nb_two)
print("%s Verbatims ont été classé dans 3 classes" % nb_three)
print("%s Verbatims ont été classé dans 4 classes" % nb_four)
print("")
print ("------------------------------------------------")
print("")
print("Temps d'execution final :" )
print(time.time() - start_time)