import urllib
from bs4 import BeautifulSoup
import pandas
import nltk
#nltk.download('punkt')
import csv
import tqdm
import re
import numpy as np

righe=1000
#dataset con i comportamenti degli users
tsv_file = open("behaviors.tsv")
read_tsv = pandas.read_csv(tsv_file, sep="\t", header=None, names=["IID", "UID", "Time", "History", "Imp"], usecols=[1,3])
tsv_file.close()

B=pandas.DataFrame(read_tsv)
read_tsv.info(null_counts=True)
C=B.dropna()
C=C.reset_index(drop=True)
C.info(null_counts=True)

comp=C[0:righe]
Hist=[]
Storie=[]
for t in tqdm.tqdm(range(len(comp))):
    a=comp.History[t].split(" ")
    Hist.append(a)
    Storie.extend(a)

S_norep = list(dict.fromkeys(Storie))


#dataset con i dati riguardanti gli items
news_file=open("news.tsv", encoding="Latin1")
read_news=pandas.read_csv(news_file, sep="\t", header=None, names=["ID", "Categoria", "SubCategoria", "Titolo", "Abstract", "URL", "TE", "AE"], usecols=[0, 1, 2, 3, 5])
read_news.info(null_counts=True)
news=pandas.DataFrame(read_news)
print(news)

news.loc[news["ID"] == "N113363"] #rimuove articolo senza url
news=news.drop(46236)

news_file.close()

news2={}
news2=pandas.DataFrame(news2)
for i in tqdm.tqdm(range(0,len(S_norep))): 
    a=news.loc[news["ID"] == S_norep[i]] #prende articoli contenuti in Snorep
    news2=pandas.concat([news2, a], ignore_index=True) #nuovo dataset

# print(news2)
# news2.info(null_counts=True)


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
with open("testi.csv", "w") as file:
     writer=csv.writer(file)
     for i in tqdm.tqdm(range(0, len(news2))):
         url=news2.URL[i]
         html = urllib.request.urlopen(url)
         testo=vattene(html)
         writer.writerow([news2.ID[i], testo]) 
        

#apertura file testi
read = pandas.read_csv("testi.csv", names=["ID", "Testo"], header=None, error_bad_lines=False) 
print(read.head())
read.info()


########PREPROCESSING DEI TESTI

#estraiamo le parole dai testi: le salviamo in "parole", una lista di lista di liste di stringhe
parole=[]
for t in range(0,len(read)):
    parole.append(read.Testo[t].split(" "))
    

#RIMOZIONE DELLE STOPWORDS   
from nltk.corpus import stopwords

#nltk.download('stopwords')
stop_words= set(stopwords.words("english"))
    
lettere = list('abcdefghijklmnopqrstuvwxyz')
numeri = list('0123456789')

for i in range(0,len(lettere)):
    stop_words.add(lettere[i])
for i in range(0,len(numeri)):
    stop_words.add(numeri[i])

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
#nltk.download('wordnet')
lem = WordNetLemmatizer()

#nltk.download('averaged_perceptron_tagger')
def preprocessing(parole):
    texts=[]
    for i in tqdm.tqdm(range(len(parole))):
        #tutto in minuscolo
        minuscolo=[]
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

texts=preprocessing(parole)    

# per salvare i testi processati in un file (UTILE PER QUANDO SAREMO SICURE SUL 
#PREPROCESSING)
# with open("testi_processati.csv", "w") as file:
#     writer=csv.writer(file)
#     for i in range(len(S_norep)):
#         writer.writerow([S_norep[i],  texts[i]])

#############################FINE PREPROCESSING DEI DATI
    

####wordcloud per un controllo visivo
import matplotlib.pyplot as plt
def plot_cloud(wordcloud):  
    # Set figure size
    plt.figure(figsize=(40, 30))
    # Display image
    plt.imshow(wordcloud) 
    # No axis details
    plt.axis("off");

from wordcloud import WordCloud
# Generate word cloud

stringa_text=str(texts[1])
import string

out = stringa_text.replace("'","")
out1=out.replace(",","")
out2=out1.replace("]","")
out3=out2.replace("[","")

wordcloud = WordCloud(width = 3000, height = 2000, random_state=1, 
                      collocations=False).generate(out3)
# Plot
plot_cloud(wordcloud)
    


######## TF.IDF

def CountFreq(word_list):
    word_dict={} 
    for word in word_list:
        if word not in word_dict:
            word_dict[word]=1
        else:
            word_dict[word]+=1
    return word_dict
    

# n_{i,j} numero di occorrenze del termine i nel documento j 
# max n_{i,j} numero di occorrenze massimo nel documento j
#|D| è il numero di documenti nella collezione
#|{d: i in d}| numero di documenti che contengono il termine i
 
      

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
            tf=tot_doc[j][k[i]]/max_f
            idf=np.log10(N/n_i[k[i]]) #n_i[k[i]] n documenti che contengono termine i
            tfidf_doc.append([k[i], tf*idf])
        tfidf_corpus.append(tfidf_doc)
    return tfidf_corpus   
      
x=TF_IDF(texts)
    
    
###########################LDA (CON gensim)
  
# Count word frequencies
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1
# Only keep words that appear more than once
processed_corpus = [[token for token in text if frequency[token] > 1] 
                    for text in texts]
#we want to associate each word in the corpus with a unique integer ID
# We can do this using the gensim.corpora.Dictionary class. This dictionary defines the 
#vocabulary of all words that our processing knows about
from gensim import corpora, models
dictionary=corpora.Dictionary(processed_corpus)

#modo per creare a mano il dizionario (attenzione: qui sono state lasciate le parole 
# con frequenza pari a 1)----> IMPLEMENTARE FUNZIONE CHE RIMUOVE LE PAROLE CON FREQ=1
# tutti=[]
# for i in tqdm.tqdm(range(len(texts))):
#     tutti=tutti+texts[i]
    
# dizionario_corpus = list(dict.fromkeys(tutti))

#rappresentazione tramite vettori dei documenti
corpus=[dictionary.doc2bow(text) for text in processed_corpus]

import pickle 
filename = 'lda_model.sav'
ldamodel=models.ldamodel.LdaModel(corpus, num_topics=100, id2word=dictionary, passes=20)
pickle.dump(ldamodel, open(filename, 'wb')) #per salvare il modello su file

# load the model from disk...basta caricare questo file!!
ldamodel = pickle.load(open('lda_model.sav', 'rb'))

from pprint import pprint
#pretty-print (si capisce meglio)
pprint(ldamodel.print_topics())

#ottengo la rappresentazione in dimesioni latenti di tutti i testi del corpus
doc_lda = ldamodel[corpus]

############CONTENT BASED PROFILES
     
                          
def ContentBasedProfile(Hist_0, doc_lda):
    #dividere
    testi=[]#ctesti (rappresentati tramite lda) delle news che l'utente in questione
    #ha letto
    for i in range(len(doc_lda)):
        if S_norep[i] in Hist_0:
            testi.append(doc_lda[i])
    dimensioni=[] #lista di dimensioni (riferita a tutte le news salvate in testi)
    pesi=[] #lista dei pesi corrispondenti
    for i in range(len(testi)):
        t=list(zip(*testi[i])) #separo le tuple(dimensioni, peso)
        dimensioni.extend(list(t[0]))    
        pesi.extend(list(t[1]))
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
    return somme
    
#applico la funzione a tutti gli user
u_profile_lda=[]
for i in tqdm.tqdm(range(len(Hist))):
    u_profile_lda.append(ContentBasedProfile(Hist[i],doc_lda))






    
    




        