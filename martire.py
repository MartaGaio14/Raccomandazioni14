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
righe=1000 #numero di utenti che verranno campionati
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
    a=len(Hist[i])//2
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

        
#########################FILE CON LE INFORMAZIONI SULLE NEWS: news_test######################### 

######## apertura del file
news_file=open("news_test.tsv", encoding="Latin1")
read_news=pandas.read_csv(news_file, sep="\t", header=None, names=["ID", "Categoria", "SubCategoria", "Titolo", "Abstract", "URL", "TE", "AE"], usecols=[0, 5])
read_news=read_news.dropna()
read_news.info(null_counts=True)
news=pandas.DataFrame(read_news)
news_file.close()

######## funzione per l'estrazione del testo
def vattene(html):
    soup = BeautifulSoup(html) # crea oggetto bs4 dal link html
    # remove javascript and stylesheet code
    for script in soup(["script", "style"]): 
        script.extract()
    text = soup.get_text()
    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines()) #crea generatore
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)
    return text

######## per il DATASET DI TRAINING:
##dal dataset completo selezioniamo solo le righe contenenti le news che sono in S_norep
news2={}
news2=pandas.DataFrame(news2)
for i in tqdm.tqdm(range(0,len(S_norep))): 
    a=news.loc[news["ID"] == S_norep[i]] #prende articoli contenuti in Snorep
    news2=pandas.concat([news2, a], ignore_index=True) #nuovo dataset
##estrazione dei testi delle news
with open("testi_train.csv", "w") as file:
     writer=csv.writer(file)
     for i in tqdm.tqdm(range(0, len(news2))):
         url=news2.URL[i]
         html = urllib.request.urlopen(url)
         testo=vattene(html)
         writer.writerow([news2.ID[i], testo]) 

######## per il DATASET DI TEST:
##dal dataset completo selezioniamo solo le righe contenenti le news che sono in S_norep2
news3={}
news3=pandas.DataFrame(news3)
for i in tqdm.tqdm(range(0,len(S_norep2))): 
    a=news.loc[news["ID"] == S_norep2[i]] #prende articoli contenuti in Snorep
    news3=pandas.concat([news3, a], ignore_index=True) #nuovo dataset   
##estrazione dei testi delle news
with open("testi_test.csv", "w") as file:
     writer=csv.writer(file)
     for i in tqdm.tqdm(range(0, len(news3))):
         url=news3.URL[i]
         html = urllib.request.urlopen(url)
         testo=vattene(html)
         writer.writerow([news3.ID[i], testo]) 
        
######## apertura file testi
         
#dataset di training
testi_train = pandas.read_csv("testi_train.csv", names=["ID", "Testo"], header=None, error_bad_lines=False) 

#dataset di test
testi_test = pandas.read_csv("testi_test.csv", names=["ID", "Testo"], header=None, error_bad_lines=False) 




############################# Lavoriamo sul DATASET DI TRAINING##################

######## Preprocessing dei testi delle news di training

inizio=time.time()
N_CPU = mp.cpu_count()
pool=mp.Pool(processes=N_CPU)
texts=pool.map(preprocessing, list(testi_train.Testo))
pool.close()
pool.join()
fine=time.time()
print(fine-inizio)



######## Rappresentazione in LDA per le news di training

##creazione del corpus
#frequenza di ogni parola
frequency = defaultdict(int) 
for text in texts:
    for token in text:
        frequency[token] += 1
#teniamo solo le parole che si ripetono più di una volta
processed_corpus = [[token for token in text if frequency[token] > 1] 
                    for text in texts]
#a ogni parola associamo un numero
dictionary=corpora.Dictionary(processed_corpus)
#a ogni numero corrispondente alle parole si associa la frequenza
corpus=[dictionary.doc2bow(text) for text in processed_corpus]

##alleniamo il modello e lo salviamo su file... PARTE DA NON FAR GIRARE
filename = 'lda_model_porter.sav'
ldamodel=models.ldamodel.LdaModel(corpus, num_topics=100, id2word=dictionary, passes=20)
pickle.dump(ldamodel, open(filename, 'wb')) #per salvare il modello su file

#carichiamo il file col modello allenato
ldamodel = pickle.load(open('lda_model.sav', 'rb'))

##appresentazione in dimesioni latenti di tutti i testi del corpus

doc_lda = ldamodel[corpus] #lista di liste
lda_dict=[] #lista di dizionari (utile per risultati)
for i in tqdm.tqdm(range(len(doc_lda))):
    lda_dict.append(dict(doc_lda[i])) 
    
######## Rappresentazione in TFIDF per le news di training
doc_tfidf=TF_IDF(texts)

######## Content based profile

#in rappresentazione lda
u_profile_lda=utenti_lda(Hist,doc_lda, S_norep) 
#in rappresentazione tfidf
u_profile_tfidf=utenti_tfidf_par(Hist,doc_tfidf, S_norep)




u_profile_tfidf2[999]
len(u_profile_tfidf)
############################# Lavoriamo sul DATASET DI TEST#####################

######## Preprocessing dei testi delle news di test

texts2=preprocessing(testi_test)  


######## Rappresentazione in LDA per le news di test

##creazione del corpus
#frequenza di ogni parola
frequency2 = defaultdict(int) 
for text in texts:
    for token in text:
        frequency2[token] += 1
#teniamo solo le parole che si ripetono più di una volta
processed_corpus2 = [[token for token in text if frequency2[token] > 1] 
                    for text in texts]
#a ogni parola associamo un numero
dictionary2=corpora.Dictionary(processed_corpus2)
#a ogni numero corrispondente alle parole si associa la frequenza
corpus2=[dictionary2.doc2bow(text) for text in processed_corpus2]

##appresentazione in dimesioni latenti di tutti i testi del corpus

doc_lda2 = ldamodel[corpus2] #lista di liste

lda_dict2=[] #lista di dizionari
for i in tqdm.tqdm(range(len(doc2_lda))):
    lda_dict2.append(dict(doc2_lda[i]))


######## Rappresentazione in TFIDF per le news di training
doc2_tfidf=TF_IDF(texts2)


         
############################# RACCOMANDAZIONI#####################################

#crea vettori di pesi per utenti e news corrispondenti ad uno stesso termine (chiave) 

with open("risultati.csv", "w") as file:
     writer=csv.writer(file)
     #for i in tqdm.tqdm(range(len(u_profile_lda))): #gira sui 1000 utenti
     for i in tqdm.tqdm(range(10)): #gira sui 1000 utenti
         for j in range(len(lda_dict2)): #gira sulle nuove news
             u=Id_utente[i]
             n=S_norep2[j]
             #s_lda=similarità(u_profile_lda[i], lda_dict2[j])
             s_tfidf=similarità(u_profile_tfidf[i],doc2_tfidf[j])
             writer.writerow([u,n, s_lda])
             
risultati=pandas.read_csv("risultati.csv", names=["UID","NID", "LDA"], header=None, error_bad_lines=False)    
    


##########PRECISIONE RECALL E FALSE POSITIVE RATE
N=10

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



