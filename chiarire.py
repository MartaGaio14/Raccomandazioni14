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

print(news2)
news2.info(null_counts=True)


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
# reg=re.compile(r"[\w]+")

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
    

# n_{i,j}} numero di occorrenze del termine i nel documento j 
#|d_{j}|} numero di termini del documento 
#|D| è il numero di documenti nella collezione
#|{d: i in d}| numero di documenti che contengono il termine i  

def TF_IDF(texts):
    tot_doc=[]  
    for i in range(0, len(texts)): 
        b=CountFreq(texts[i])
        tot_doc.append(b) #lista di dizionari freq per ogni documento
    N=len(tot_doc)
    tfidf_corpus = []    
    for j in tqdm.tqdm(range(0, len(tot_doc))):
        k=list(tot_doc[j].keys())
        tfidf_doc={}
        for i in range(0, len(tot_doc[j])):
            tf=tot_doc[j][k[i]]/len(tot_doc[j])
            count=0
            for z in range(0, len(tot_doc)):
                if k[i] in tot_doc[z]:
                    count+=1
            idf=np.log(N/count)
            tfidf_doc[k[i]]= tf*idf
        tfidf_corpus.append(tfidf_doc)
    return tfidf_corpus   
  
    
  

def TF_IDF(texts):
    tot_doc=[]
    tot_words=[]
    for i in range(0, len(texts)): 
        b=CountFreq(texts[i])
        tot_doc.append(b) #lista di dizionari freq per ogni documento
        tot_words.extend(texts[i])
    tot_words=np.unique(tot_words)
    indexes=np.unique(tot_words, return_index=True)[1]
    [tot_words[index] for index in sorted(indexes)]
    
    for x.askeys() in tqdm.tqdm(tot_words):
        count=0
        for j in range(0, len(tot_doc)):
            if x in tot_doc[j]:
                count+=1
    
    N=len(tot_doc)
    tfidf_corpus = []    
    for j in tqdm.tqdm(range(0, len(tot_doc))):
        k=list(tot_doc[j].keys())
        tfidf_doc={}
        for i in range(0, len(tot_doc[j])):
            tf=tot_doc[j][k[i]]/len(tot_doc[j])
            count=0
            for z in range(0, len(tot_doc)):
                if k[i] in tot_doc[z]:
                    count+=1
            idf=np.log(N/count)
            tfidf_doc[k[i]]= tf*idf
        tfidf_corpus.append(tfidf_doc)
    return tfidf_corpus   

x=TF_IDF(texts)


from gensim import corpora, models
dictionary = corpora.Dictionary(texts)
bow_corpus = [dictionary.doc2bow(doc) for doc in texts]
tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]
len(corpus_tfidf)
from pprint import pprint
for doc in corpus_tfidf:
    pprint(doc)
    break      

corpus_tfidf[0]
x[0]
    
    
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
ldamodel=models.ldamodel.LdaModel(corpus, num_topics=100, id2word=dictionary, passes=20)

from pprint import pprint
#pretty-print (si capisce meglio)
pprint(ldamodel.print_topics())

#ottengo la rappresentazione in dimesioni latenti di tutti i testi del corpus
doc_lda = ldamodel[corpus]
pprint(doc_lda[1])

#vediamo quante dimensioni latenti hanno pesi diversi da zero per ogni articolo
lunghezze=[]
for t in doc_lda:
    lunghezze.append(len(t))
    
import matplotlib.pyplot as plt
plt.plot(lunghezze)
###ne basterebbero 35!

##salvo la rappresentazione in dimensioni latenti in un file csv
with open("testi_lda.csv", "w") as file:
    writer=csv.writer(file)
    for i in range(len(S_norep)):
        writer.writerow([S_norep[i],  doc_lda[i]])

#apertura del file csv con idnews+rappresentazione  dimensioni latenti
csv_file = open("testi_lda.csv", encoding="Latin1")
LDA_OUT = pandas.read_csv(csv_file, sep=",", header=None, names=["ID", "Testo_LDA"])
csv_file.close()





    
    




csv_file.close()
        