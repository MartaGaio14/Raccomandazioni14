import urllib
from bs4 import BeautifulSoup
import pandas
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import csv
import tqdm
import re
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from collections import defaultdict
from gensim import corpora, models
import pickle



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

S_norep.remove("N113363")

for i in range(len(Hist)):
    if Hist[i].count("N113363")>0:
        Hist[i].remove("N113363")
#rimuovo anche questo che è nel testing set
for i in range(len(Hist)):
    if Hist[i].count("N110434")>0: 
        Hist[i].remove("N110434")     


  
#dataset con i dati riguardanti gli items
news_file=open("news_test.tsv", encoding="Latin1")
read_news=pandas.read_csv(news_file, sep="\t", header=None, names=["ID", "Categoria", "SubCategoria", "Titolo", "Abstract", "URL", "TE", "AE"], usecols=[0, 5])
read_news.info(null_counts=True)
news=pandas.DataFrame(read_news)
#print(news)
news_file.close()

news.loc[news["ID"] == "N113363"] #rimuove articolo senza url
news.loc[news["ID"] == "N110434"] 
news=news.drop(46027)
news=news.drop(4694)

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



########PREPROCESSING DEI TESTI

#RIMOZIONE DELLE STOPWORDS   

#nltk.download('stopwords')
stop_words= set(stopwords.words("english"))
    
lettere = list('abcdefghijklmnopqrstuvwxyz')
numeri = list('0123456789')


for i in range(0,len(lettere)):
    stop_words.add(lettere[i])
for i in range(0,len(numeri)):
    stop_words.add(numeri[i])
stop_words.add("getty")

#PART OF SPEACH TAGGING
REM=['CC','CD','DT','EX','IN','LS','MD','PDT','POS','PRP','PSRP$','RB',
     'RBR','RBS','TO','UH','WDT','WP','WPD$','WRB']

#la funzione restituisce un vettore lungo come la lista di tuple con 0 se la tupla è da tenere e 1 
#se è da togliere
def eliminare(tagged_words1):
    togli=np.zeros(len(tagged_words1))
    res = list(zip(*tagged_words1)) #zippiamo la lista di tuple
    res=res[1] #prendiamo solo i tag
    for i in np.arange(len(tagged_words1)):  
        #il ciclo prende nota di quali sono le parole da eliminare 
        tup=res[i]
        for j in REM:
            if tup==j:
                togli[i]=1
    return togli

#LEMMING
from nltk.stem.wordnet import WordNetLemmatizer
lem = WordNetLemmatizer()

def preprocessing(testi_file):
    parole=[] #parole estratte dai testi, lista di lista di liste di stringhe
    texts=[]
    for i in tqdm.tqdm(range(len(testi_file))):
        parole.append(testi_file.Testo[i].split(" "))
        minuscolo=[] #tutto in minuscolo
        for j in range(0,len(parole[i])):
            minuscolo.append(parole[i][j].lower())
        #rimozione delle stopwords
        words_nostop=[word for word in minuscolo if word not in stop_words]
        #rimozione della punteggiatura
        words_nopunct= [word for word in words_nostop if word.isalnum()] 
        #part of speach tagging
        tagged_words=nltk.pos_tag(words_nopunct) 
        togli=eliminare(tagged_words)
        togli= np.array(togli, dtype=int)
        finali=list(np.array(words_nopunct)[togli==0])
        #lemming
        lemmed_words=[]
        for w in finali:
            lemmed_words.append(lem.lemmatize(w,"v"))
        finali=lemmed_words
        #salvo le parole rimaste per ogni documento in una lista di liste
        texts.append(finali)
    return texts

texts=preprocessing(testi_train)    

#############################FINE PREPROCESSING DEI DATI


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
filename = 'lda_model.sav'
ldamodel=models.ldamodel.LdaModel(corpus, num_topics=100, id2word=dictionary, passes=20)
pickle.dump(ldamodel, open(filename, 'wb')) #per salvare il modello su file

# load the model from disk...basta caricare questo file!!
ldamodel = pickle.load(open('lda_model.sav', 'rb'))

#ottengo la rappresentazione in dimesioni latenti di tutti i testi del corpus
doc_lda = ldamodel[corpus]
lda_dict=[] #lista di dizionari
for i in tqdm.tqdm(range(len(doc_lda))):
    lda_dict.append(dict(doc_lda[i])) #topic per ogni articolo
    
######## TF.IDF

def CountFreq(word_list):
    word_dict={} 
    for word in word_list:
        if word not in word_dict:
            word_dict[word]=1
        else:
            word_dict[word]+=1
    return word_dict  

def TF_IDF(texts):
    tot_doc=[] #lista di dizionari freq per ogni documento
    allwords=[]#lista parole singole per ogni documento 
    for z in range(0, len(texts)): 
        b=CountFreq(texts[z])
        tot_doc.append(b) 
        norep=np.unique(texts[z])
        allwords.extend(norep)    
    n_i=CountFreq(allwords) #numero di documenti che contengono un termine 
    N=len(tot_doc) #numero di documenti nel corpus
    tfidf_corpus = []
    for j in tqdm.tqdm(range(0, len(tot_doc))):
        k=list(tot_doc[j].keys())
        tfidf_doc=[]
        for i in range(0, len(tot_doc[j])):
            max_f=max(list(tot_doc[j].values())) #parola con massima freq nel documento j
            tf=tot_doc[j][k[i]]/max_f #numero di occorrenze del termine i nel documento j/max_f
            idf=np.log10(N/n_i[k[i]]) #n_i[k[i]] n documenti che contengono termine i
            tfidf_doc.append([k[i], tf*idf])
        tfidf_corpus.append(dict(tfidf_doc))
    return tfidf_corpus   
      
doc_tfidf=TF_IDF(texts)
    
    
############CONTENT BASED PROFILES
def ContentBasedProfile(Hist_0, dimensioni, pesi):
    somme=[] #lista di liste di due elementi ciascuna: numero del topic + somma dei 
    #pesi corrispondenti
    b=[] #lista delle dimensioni già viste
    for i in range(len(dimensioni)):
        if dimensioni[i] not in b:
            b.append(dimensioni[i]) 
            a=[dimensioni[i], pesi[i]] 
            somme.append(a)
        else:
            for j in range(len(b)):
                if dimensioni[i]==b[j]:
                    somme[j][1]+=pesi[i]
    for s in range(len(somme)):
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
    # if len(profilo)>4000:
    #     profilo=dict(sorted(profilo.items(), key=lambda item: item[1])[0:4000])
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
    




#################################DATASET DI TESTING#################################

Storie_test=[] #lista di tutte le news lette
for i in range(len(n_test)):
    Storie_test.extend(n_test[i])

#lista delle 11733 news per il test set            
S_norep2 = list(dict.fromkeys(Storie_test))
S_norep2.remove("N110434")#articolo senza url
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
def compara(dictU, dictA):
    utente=[]
    articolo=[]
    for keyU in dictU:
        for keyA in dictA:
            if keyU == keyA:
                utente.append(dictU[keyA])
                articolo.append(dictA[keyA])
    return utente,articolo
            
def CosDist(u, v):
    dist = np.dot(u, v) / (LA.norm(u) * LA.norm(v))
    return dist

def similarità(dictU, dictA):
    (a,b)=compara(dictU, dictA)
    return CosDist(a,b)


with open("risultati.csv", "w") as file:
     writer=csv.writer(file)
     for i in tqdm.tqdm(range(len(u_profile_lda))): #gira sugli utenti
         for j in range(len(lda_dict2)): #gira sulle nuove news
             u=Id_utente[i]
             n=S_norep2[j]
             #s_tfidf=similarità(u_profile_tfidf[i],doc2_tfidf[j])
             s_lda=similarità(u_profile_lda[i], lda_dict2[j])
             writer.writerow([u,n,s_lda])
             
risultati=pandas.read_csv("risultati.csv", names=["UID","NID", "LDA"], header=None, error_bad_lines=False)    
    

    
##########CURVA ROC
N=10

precision=recall=fp_rate=0
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
    fp=N-tp #false positive
    fn=len(n_test[u])-tp #false negative
    tn=len(S_norep2)-len(n_test[u])-fp #true negative 
    precision+=tp/(tp+fp)
    recall+=tp/(tp+fn)
    fp_rate+=fp/(fp+tn)
    inizio=fine+1
    fine=inizio+len(S_norep2)


def ROC

