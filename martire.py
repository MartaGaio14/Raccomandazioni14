import urllib
from bs4 import BeautifulSoup
import pandas
import csv
import tqdm
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from collections import defaultdict
from gensim import corpora, models
import pickle

#importiamo i moduli da noi creati
from preprocessing import *
from tfidf import *
from similarita import similarità


#apro il file riguardante gli utenti (Test Set)
righe=1000
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
    
    
#eliminiamo le news poblematiche
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
        
#divisione training e test set
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

        
########################################DATASET DI TRAINING######################### 
Storie_train=[] #lista di tutte le news lette
for i in range(len(n_training)):
    Storie_train.extend(n_training[i])
            
S_norep = list(dict.fromkeys(Storie_train))

  
#dataset con i dati riguardanti gli items
news_file=open("news_test.tsv", encoding="Latin1")
read_news=pandas.read_csv(news_file, sep="\t", header=None, names=["ID", "Categoria", "SubCategoria", "Titolo", "Abstract", "URL", "TE", "AE"], usecols=[0, 5])
read_news=read_news.dropna()
read_news.info(null_counts=True)
news=pandas.DataFrame(read_news)
#print(news)
news_file.close()

news2={}
news2=pandas.DataFrame(news2)
for i in tqdm.tqdm(range(0,len(S_norep))): 
    a=news.loc[news["ID"] == S_norep[i]] #prende articoli contenuti in Snorep
    news2=pandas.concat([news2, a], ignore_index=True) #nuovo dataset

#news2.info()



#estrazione del testo
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

    
# file testi estratti (titolo compreso)
with open("testi_train.csv", "w") as file:
     writer=csv.writer(file)
     for i in tqdm.tqdm(range(0, len(news2))):
         url=news2.URL[i]
         html = urllib.request.urlopen(url)
         testo=vattene(html)
         writer.writerow([news2.ID[i], testo]) 
        

#apertura file testi
testi_train = pandas.read_csv("testi_train.csv", names=["ID", "Testo"], header=None, error_bad_lines=False) 
testi_train.info()


## preprocessing delle news di training
texts=preprocessing(testi_train)    


###########################LDA (CON gensim)
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1
# Only keep words that appear more than once
processed_corpus = [[token for token in text if frequency[token] > 1] 
                    for text in texts]
dictionary=corpora.Dictionary(processed_corpus)

corpus=[dictionary.doc2bow(text) for text in processed_corpus]
filename = 'lda_model_porter.sav'
ldamodel=models.ldamodel.LdaModel(corpus, num_topics=100, id2word=dictionary, passes=20)
pickle.dump(ldamodel, open(filename, 'wb')) #per salvare il modello su file

# load the model from disk...basta caricare questo file!!
ldamodel = pickle.load(open('lda_model.sav', 'rb'))

#ottengo la rappresentazione in dimesioni latenti di tutti i testi del corpus
doc_lda = ldamodel[corpus]
lda_dict=[] #lista di dizionari
for i in tqdm.tqdm(range(len(doc_lda))):
    lda_dict.append(dict(doc_lda[i])) #topic per ogni articolo
    
######## TF.IDF sulle 


      
doc_tfidf=TF_IDF(texts)

############CONTENT BASED PROFILES

def ContentBasedProfile(Hist_0, dimensioni, pesi):
    #lista di liste di due elementi ciascuna: numero del topic+somma dei pesi corrispondenti
    somme=[] 
    b=[] #lista delle dimensioni già viste
    for i in range(len(dimensioni)):
        if dimensioni[i] not in b:
            b.append(dimensioni[i]) 
            a=[dimensioni[i], pesi[i]] 
            somme.append(a)
        else:
            for j in range(len(b)):
                if dimensioni[i]==b[j]:
                    somme[j][1]+=pesi[i] #somma al peso corrispondente alla dimensione
    for s in range(len(somme)):
         #divido pesi relativi a ciascuna parola per il n di news lette da ciascun utente
        p=somme[s][1]/len(Hist_0) 
        somme[s][1]= p
    return dict(somme)


####################CONTENT BASED PROFILE IN RAPPRESENTAZIONE TFIDF

u_profile_tfidf=[]#lista di dizionari, uno per ogni utente
for i in tqdm.tqdm(range(len(Hist))): #i gira negli user
    testi=[]#lista di dizionari, ogni dizionario è una news in tfidf letta dall'utente
    #in questione (l'iesimo)
    for j in range(len(doc_tfidf)):
        if S_norep[j] in Hist[i]:
            testi.append(doc_tfidf[j])
    diz={}
    #dimensioni-->lista di dimensioni (riferita a tutte le news salvate in testi)
    #pesi-->lista dei pesi corrispondenti (i pesi sono le chiavi del dizionario)
    for t in range(len(testi)):
        pesi=list(testi[t].values())
        dimensioni=list(testi[t].keys())
        for j in range(len(testi[t])):
            diz[dimensioni[j]]=pesi[j]
    profilo=dict(ContentBasedProfile(Hist[i],list(diz.keys()),list(diz.values())))
    if len(profilo)>1000:
        profilo=dict(sorted(profilo.items(), key=lambda item: item[1])[0:1000])
    u_profile_tfidf.append(profilo)
    
    
###################CONTENT BASED PROFILE IN RAPPRESENTAZIONE LDA
    
#applico la funzione a tutti gli user 
u_profile_lda=[]
for i in tqdm.tqdm(range(len(Hist))): #i gira negli user   
    testi=[]
    for j in range(len(doc_lda)):
        if S_norep[j] in Hist[i]:
            testi.append(doc_lda[j])
    dimensioni=[] #lista di dimensioni (riferita a tutte le news salvate in testi)
    pesi=[] #lista dei pesi corrispondenti
    for i in range(len(testi)):
        t=list(zip(*testi[i])) #separo le tuple(dimensioni, peso)
        dimensioni.extend(list(t[0]))    
        pesi.extend(list(t[1]))    
    u_profile_lda.append(ContentBasedProfile(Hist[i],dimensioni, pesi))
    #creata una lista di dizionari
 
    

#################################DATASET DI TESTING#################################

Storie_test=[] #lista di tutte le news lette
for i in range(len(n_test)):
    Storie_test.extend(n_test[i])

#lista delle 11733 news per il test set            
S_norep2 = list(dict.fromkeys(Storie_test))

#dataset con i dati riguardanti gli items
news3={}
news3=pandas.DataFrame(news3)
for i in tqdm.tqdm(range(0,len(S_norep2))): 
    a=news.loc[news["ID"] == S_norep2[i]] #prende articoli contenuti in Snorep
    news3=pandas.concat([news3, a], ignore_index=True) #nuovo dataset
#estraiamo i testi e li salviamo su file

with open("testi_test.csv", "w") as file:
     writer=csv.writer(file)
     for i in tqdm.tqdm(range(0, len(news3))):
         url=news3.URL[i]
         html = urllib.request.urlopen(url)
         testo=vattene(html)
         writer.writerow([news3.ID[i], testo]) 

testi_test = pandas.read_csv("testi_test.csv", names=["ID", "Testo"], header=None, error_bad_lines=False) 
texts2=preprocessing(testi_test)


frequency = defaultdict(int) 
for text in texts2:
    for token in text:
        frequency[token] += 1 #conta frequenze
# Only keep words that appear more than once
processed_corpus2 = [[token for token in text if frequency[token] > 1] 
                    for text in texts2]
dictionary2=corpora.Dictionary(processed_corpus2) #associa a termini in processed_corpus un nuovo codice 
corpus2=[dictionary2.doc2bow(text) for text in processed_corpus2] #dizionario codice parola e frequenza
doc2_lda = ldamodel[corpus2]
lda_dict2=[] #lista di dizionari
for i in tqdm.tqdm(range(len(doc2_lda))):
    lda_dict2.append(dict(doc2_lda[i])) #topic per ogni articolo

doc2_tfidf=TF_IDF(texts2) # TF-IDF

         
##########RACCOMANDAZIONI
#crea vettori di pesi per utenti e news corrispondenti ad uno stesso termine (chiave) 


with open("risultati_tf.csv", "w") as file:
     writer=csv.writer(file)
     for i in tqdm.tqdm(range(500,len(u_profile_lda))): #gira sui 1000 utenti
         for j in range(len(lda_dict2)): #gira sulle nuove news
             u=Id_utente[i]
             n=S_norep2[j]
             s_lda=similarità(u_profile_lda[i], lda_dict2[j])
             s_tfidf=similarità(u_profile_tfidf[i],doc2_tfidf[j])
             writer.writerow([u,n, s_lda, s_tfidf])
             
risultati=pandas.read_csv("risultati_lda.csv", names=["UID","NID", "LDA"], header=None, error_bad_lines=False)    
    

    
##########CURVA ROC
N=100

precision=recall=fp_rate=0
tpli=[]
inizio=0
fine=len(S_norep2)
for u in tqdm.tqdm(range(len(n_test[0:100]))):
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



