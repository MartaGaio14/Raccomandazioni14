righe=1000 #numero di utenti da campionare
Ntfidf=750 #lunghezza massima della rappresentazione in tfidf
train_test=0.8 #percentuale di news di ogni history da mettere nel dataset di training
N=10 #numero di news da raccomandare

##nome file dataset delle news coi body estratti
filename_body="testi1000.csv"

##nome file modello lda (da scrivere)
filename_lda = 'lda_model_snow2000_80.sav'

##nome file risultati similarità (da scrivere)
filename_sim="risultati2000_80.csv"

import urllib
from bs4 import BeautifulSoup
import pandas
import csv
import tqdm
import numpy as np
from collections import defaultdict
from gensim import corpora, models
import pickle
import time

#importiamo i moduli da noi creati
from preprocessing import *
from tfidf import *
from profili_utenti import utenti_tfidf_par
from profili_utenti import utenti_tfidf
from profili_utenti import utenti_lda
from similarita import similarità


###########FILE SUL COMPORTAMENTO DEGLI UTENTI: behaviors_test######################

######## apertura del file

Set = open("behaviors_test.tsv")
Set1 = pandas.read_csv(Set, sep="\t", header=None, names=["IID", "UID", "Time", "History", "Imp"], usecols=[1, 3])
Set.close()

GrandeSet=Set1.dropna()
GrandeSet=GrandeSet.reset_index(drop=True)
GrandeSet.info(null_counts=True)

Hist=[] #lista di liste news per utente
Id_utente=[] #lista id utenti
for i in tqdm.tqdm(range(len(GrandeSet))):
    a=GrandeSet.History[i].split(" ")
    while len(Hist) < (righe):
        if len(a) > 100:
            Hist.append(a) 
            Id_utente.append(GrandeSet.UID[i])
        break
    
    
######## eliminiamo le news poblematiche
#le prime due non hanno l'url e l'ultima non è nel dataset delle news
for i in range(len(Hist)):
    if Hist[i].count("N113363")>0:
        Hist[i].remove("N113363")
for i in range(len(Hist)):
    if Hist[i].count("N110434")>0: 
        Hist[i].remove("N110434")
for i in range(len(Hist)):
    if Hist[i].count("N89741")>0: 
        Hist[i].remove("N89741")
        
######## divisione in training set e test set
n_test=[] 
n_training=[]
for i in range(len(Hist)):
    a=int(len(Hist[i])*train_test)
    temp_train=[]
    temp_test=[]
    for j in range(len(Hist[i])):
        if j < a:
            temp_train.append(Hist[i][j])
        else:
            temp_test.append(Hist[i][j])
    n_training.append(temp_train)
    n_test.append(temp_test)

##lista di tutte le news del training set che sono state lette dalla totalità
#degli utenti campionati
Storie_train=[] 
for i in range(len(n_training)):
    Storie_train.extend(n_training[i])            
S_norep = list(dict.fromkeys(Storie_train))

##lista di tutte le news del test set che sono state lette dalla totalità
#degli utenti campionati
Storie_test=[] #lista di tutte le news lette
for i in range(len(n_test)):
    Storie_test.extend(n_test[i])            
S_norep2 = list(dict.fromkeys(Storie_test))

##lista delle news totali per le quali fare il web scraping e il preprocessing
tutteNID=S_norep+S_norep2
#########################FILE CON LE INFORMAZIONI SULLE NEWS: news_test######################### 

######## apertura del file
news_file=open("news_test.tsv", encoding="Latin1")
read_news=pandas.read_csv(news_file, sep="\t", header=None, names=["ID", "Categoria", "SubCategoria", "Titolo", "Abstract", "URL", "TE", "AE"], usecols=[0, 5])
read_news=read_news.dropna()
read_news.info(null_counts=True)
news=pandas.DataFrame(read_news)
news_file.close()


##dal dataset completo selezioniamo solo le righe contenenti le news che sono in tutteNID
news2={}
news2=pandas.DataFrame(news2)
for i in tqdm.tqdm(range(0,len(tutteNID))):
    a=news.loc[news["ID"] == tutteNID[i]] #prende articoli contenuti in tutteNID
    news2=pandas.concat([news2, a], ignore_index=True) #nuovo dataset


URLS= list(news2.URL)

#with open("url_file.txt", 'w') as f:
#    for url in URLS:
#        f.write("%s\n" % url)

##estrazione del testo

from preprocessing import *

inizio=time.time()
with open (filename_body, "w", encoding="Utf-8") as file:
    writer=csv.writer(file)
    for i in tqdm.tqdm(range(len(URLS))):
        writer.writerow([news2.ID[i], extraction(URLS[i])])
fine=time.time()
print(fine-inizio)



inizio = time.time()
N_CPU = mp.cpu_count()
pool = mp.Pool(processes=N_CPU)
testi_web = pool.map(extraction, URLS[0:1000])
pool.close()
pool.join()
fine = time.time()



print("Fatto web-scraping")
######## apertura file testi
testi= pandas.read_csv("testi_news.csv", names=["ID", "Testo"], header=None, error_bad_lines=False)

####### preprocessing per tutti i testi

inizio=time.time()
N_CPU = mp.cpu_count()
pool=mp.Pool(processes=N_CPU)
texts=pool.map(preprocessing, list(testi.Testo))
pool.close()
pool.join()
fine=time.time()
print(fine-inizio)

print("Fatto preprocessing")

####### divisione dei testi processati in training e test
#i primi len(S_norep) articoli sono del dataset di training
texts_train=texts[0:len(S_norep)-1]
texts_test=texts[len(S_norep):len(texts)-1]

######## Rappresentazione in LDA per le news di training

##creazione del corpus
#frequenza di ogni parola
frequency = defaultdict(int) 
for text in texts_train:
    for token in text:
        frequency[token] += 1
#teniamo solo le parole che si ripetono più di una volta
processed_corpus = [[token for token in text if frequency[token] > 1] 
                    for text in texts_train]
#a ogni parola associamo un numero
dictionary=corpora.Dictionary(processed_corpus)
#a ogni numero corrispondente alle parole si associa la frequenza
corpus=[dictionary.doc2bow(text) for text in processed_corpus]

##alleniamo il modello e lo salviamo su file... PARTE DA NON FAR GIRARE

ldamodel=models.LdaMulticore(corpus, num_topics=100, id2word=dictionary, passes=20, workers=4)
pickle.dump(ldamodel, open(filename_lda, 'wb')) #per salvare il modello su file

print("Allenata l'LDA")
#carichiamo il file col modello allenato
#ldamodel = pickle.load(open(filename_lda, 'rb'))

##appresentazione in dimesioni latenti di tutti i testi del corpus

doc_lda = ldamodel[corpus] #lista di liste
lda_dict=[] #lista di dizionari (utile per risultati)
for i in tqdm.tqdm(range(len(doc_lda))):
    lda_dict.append(dict(doc_lda[i])) 
    
######## Rappresentazione in TFIDF per le news di training

doc_tfidf=TF_IDF(texts_train)

######## Content based profile

#in rappresentazione lda
u_profile_lda=utenti_lda(Hist,doc_lda, S_norep)
print("Calcolati profili utenti LDA")
#in rappresentazione tfidf
u_profile_tfidf=utenti_tfidf_par(Hist,doc_tfidf, S_norep)
print("Calcolati profili utenti TFIDF")

############################# Lavoriamo sul DATASET DI TEST#####################

######## Rappresentazione in LDA per le news di test

##creazione del corpus
#frequenza di ogni parola
frequency2 = defaultdict(int) 
for text in texts_test:
    for token in text:
        frequency2[token] += 1
#teniamo solo le parole che si ripetono più di una volta
processed_corpus2 = [[token for token in text if frequency2[token] > 1] 
                    for text in texts_test]
#a ogni parola associamo un numero
dictionary2=corpora.Dictionary(processed_corpus2)
#a ogni numero corrispondente alle parole si associa la frequenza
corpus2=[dictionary2.doc2bow(text) for text in processed_corpus2]

##appresentazione in dimesioni latenti di tutti i testi del corpus

doc_lda2 = ldamodel[corpus2] #lista di liste

lda_dict2=[] #lista di dizionari
for i in tqdm.tqdm(range(len(doc_lda2))):
    lda_dict2.append(dict(doc_lda2[i]))

######## Rappresentazione in TFIDF per le news di training
doc2_tfidf=TF_IDF(texts_test)

print("Calcolate lda e tfidf news di test")
############################# RACCOMANDAZIONI#####################################

#crea vettori di pesi per utenti e news corrispondenti ad uno stesso termine (chiave) 

with open(filename_sim, "w") as file:
     writer=csv.writer(file)
     for i in tqdm.tqdm(range(len(u_profile_lda))): #gira sui 1000 utenti
         for j in range(len(lda_dict2)): #gira sulle nuove news
             u=Id_utente[i]
             n=S_norep2[j]
             s_lda=similarità(u_profile_lda[i], lda_dict2[j])
             #s_tfidf=similarità(u_profile_tfidf[i],doc2_tfidf[j])
             writer.writerow([u,n, s_lda])
             
risultati=pandas.read_csv(filename_sim, names=["UID","NID", "LDA"], header=None, error_bad_lines=False)

##########PRECISIONE RECALL E FALSE POSITIVE RATE
precision=recall=fp_rate=0
tpli=[]
inizio=0
fine=len(S_norep2)
for u in tqdm.tqdm(range(len(n_test))):
    ut=risultati[inizio:fine]
    top_lda=ut.sort_values(by=['LDA'], ascending=False)[0:N]
    top_lda=top_lda.reset_index(drop=True)
    tp=0
    for i in range(len(top_lda)): 
        if top_lda.NID[i] in n_test[u]:
            tp+=1 #true positive
    tpli.append(tp)
    fp=N-tp #false positive
    fn=len(n_test[u])-tp #false negative
    tn=len(S_norep2)-len(n_test[u])-fp #true negative 
    precision+=tp/(tp+fp)
    recall+=tp/(tp+fn)
    fp_rate+=fp/(fp+tn)
    inizio=(u+1)*len(S_norep2)+1
    fine=(u+2)*len(S_norep2)

print("PRECISIONE")
print(precision/righe)



