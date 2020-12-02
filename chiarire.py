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
tsv=np.genfromtxt("behaviors.tsv", delimiter="\t", names=["IID", "UID", "Time", "History", "Imp"], usecols=[1,3], max_rows=righe, dtype=object)
Utenti=tsv["UID"].astype(str)   #lista di tutti gli id degli utenti
Art=tsv["History"].astype(str) 

Hist=[] #lista di liste (divisione utenti)
Storie=[] #lista id articoli (con ripetizione)
for i in range(0,len(tsv)):
    if Art[i] != '':
        a=Art[i].split(" ")
        Hist.append(a)
        Storie=Storie+a
  
S_norep = list(dict.fromkeys(Storie))
#crea dizionario con k=articoli (non ripetuti)


#=============================================================================
#CON PANDAS!!!!
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

for t in range(len(comp)):
    a=comp.History[t].split(" ")
    Hist.append(a)
    Storie=Storie+a
   
print(Hist)
print(Storie)

S_norep = list(dict.fromkeys(Storie))
#==============================================================================





#dataset con i dati riguardanti gli items
news_file=open("news.tsv", encoding="Latin1")
news=np.genfromtxt(news_file, delimiter="\t", names=["ID", "Categoria", "SubCategoria", "Titolo", "Abstract", "URL", "TE", "AE"], usecols=[0, 1, 2, 3, 5], dtype=object, comments=None)
news_file.close()


# News_id=news["ID"].astype(str) 
# for j in range(0,len(news)):
#     if News_id[j] == "N113363":
#         break
# news=news.drop(j)
    
news2=[]
for i in tqdm.tqdm(range(0,len(S_norep))):
    a=news[np.where(news["ID"].astype(str)==S_norep[i])] #prende articoli contenuti in S_norep
    a=list(str(a).split(", b"))
    news2.append(a)



#===================================================================
#CON PANDAS!!!!!!!!!!!

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
#==============================================================================
 

#estrazione del testo
def vattene(html):
    soup = BeautifulSoup(html) # crea oggetto bs4 dal link html
    # rimuove javascript e stylesheet code
    for script in soup(["script", "style"]): 
        script.extract()
    text = soup.get_text()
    # divide il testo in righe e rimuove spazi iniziali e finali
    lines = (line.strip() for line in text.splitlines()) #crea generatore
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)
    return text
    
with open("testi.csv", "w") as file:
    writer=csv.writer(file)
    for i in tqdm.tqdm(range(0, len(news2))):
        url=news2[i][4].replace("')]","")
        url=url.replace("'","")
        html = urllib.request.urlopen(url)
        testo=vattene(html)
        n_id=news2[i][0].replace("[(b'","")
        n_id=n_id.replace("'","")
        writer.writerow([n_id, testo])
        

#apertura file testi
csv_file = open("testi.csv", encoding="Latin1")
read = pandas.read_csv(csv_file, sep=",", header=None, names=["ID", "Testo"])
read = pandas.read_csv("testi.csv", names=["ID", "Testo"], header=None, error_bad_lines=False) #equivalente
print(read.head())

csv_file.close()

read.info()

########PREPROCESSING DEI TESTI

#estraiamo le parole dai testi: le salviamo in "parole", una lista di lista di liste di stringhe

parole=[]
for t in range(0,len(read)):
    parole.append(read.Testo[t].split(" "))
    

#RIMOZIONE DELLE STOPWORDS   
from nltk.corpus import stopwords

nltk.download('stopwords')
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
nltk.download('wordnet')
lem = WordNetLemmatizer()

nltk.download('averaged_perceptron_tagger')

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
    indici=np.zeros(int(sum(togli))) 
    togli= np.array(togli, dtype=int)
    finali=list(np.array(words_nopunct)[togli==0])
    #lemming
    lemmed_words=[]
    for w in finali:
        lemmed_words.append(lem.lemmatize(w,"v"))
    finali=lemmed_words
    #salvo le parole rimaste per ogni documento in una lista di liste
    texts.append(finali)
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
    
    
#############################LDA
  
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
#rappresentazione tramite vettori dei documenti
corpus=[dictionary.doc2bow(text) for text in processed_corpus]
ldamodel=models.ldamodel.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=20)

print(ldamodel.print_topics(num_topics=3, num_words=3))


csv_file.close()
        