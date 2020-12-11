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



#apro il file briguardante gli utenti (Test Set)
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
        
 
Storie_train=[] #lista di tutte le news lette
for i in range(len(n_training)):
    Storie_train.extend(n_training[i])
            
S_norep = list(dict.fromkeys(Storie_train))

S_norep.remove("N113363")
for i in range(len(Hist)):
    if Hist[i].count("N113363")>0:
        Hist[i].remove("N113363")     
    
    
  
#dataset con i dati riguardanti gli items
news_file=open("news_test.tsv", encoding="Latin1")
read_news=pandas.read_csv(news_file, sep="\t", header=None, names=["ID", "Categoria", "SubCategoria", "Titolo", "Abstract", "URL", "TE", "AE"], usecols=[0, 5])
read_news.info(null_counts=True)
news=pandas.DataFrame(read_news)
print(news)
news_file.close()

news.loc[news["ID"] == "N113363"] #rimuove articolo senza url
news=news.drop(46027)

news2={}
news2=pandas.DataFrame(news2)
for i in tqdm.tqdm(range(0,len(S_norep))): 
    a=news.loc[news["ID"] == S_norep[i]] #prende articoli contenuti in Snorep
    news2=pandas.concat([news2, a], ignore_index=True) #nuovo dataset

news2.info()



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
testi_news = pandas.read_csv("testi_train.csv", names=["ID", "Testo"], header=None, error_bad_lines=False) 
#read = pandas.read_csv("testi.csv", sep=",", header=None, names=["ID", "Testo"])
testi_news.info()



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
#nltk.download('wordnet')
lem = WordNetLemmatizer()

#nltk.download('averaged_perceptron_tagger')
def preprocessing(testi_file):
    parole=[] #parole estratte dai testi, lista di lista di liste di stringhe
    texts=[]
    for i in tqdm.tqdm(range(len(testi_file))):
        parole.append(testi_file.Testo[t].split(" "))
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



