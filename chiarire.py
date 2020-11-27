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


Hist=[]
Storie=[]
for i in range(0,len(tsv)):
    Hist.append(str(tsv[i][1]))
    if tsv[i][1] is not b'':
        a.append(True)
    else:
        a.append(False)
        
#stavo cercando di costruire Hist e Storie direttamente ma non ho finito
#e non so se abbia senso ahah

for t in range(len(tsv)):
    a=tsv[t][1].split("\s+")
    Hist.append(a)
    Storie=Storie+a
   
print(Hist)
print(Storie)

S_norep = list(dict.fromkeys(Storie))


np.isnan(tsv[987][1])
for i in tqdm.tqdm(range(0,len(tsv))):
    b=np.isnan(tsv[i][1])

nan_array = np.isnan(array1)
not_nan_array = ~ nan_array
array2 = array1[not_nan_array]

   

    
    


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






#dataset con i dati riguardanti gli items
news_file=open("news.tsv", encoding="Latin1")
read_news=pandas.read_csv(news_file, sep="\t", header=None, names=["ID", "Categoria", "SubCategoria", "Titolo", "Abstract", "URL", "TE", "AE"], usecols=[0, 1, 2, 3, 5])
read_news.info(null_counts=True)
news=pandas.DataFrame(read_news)
print(news)

news.loc[news["ID"] == "N113363"]
news=news.drop(46236)

news_file.close()


news2={}
news2=pandas.DataFrame(news2)
for i in tqdm.tqdm(range(0,len(S_norep))): 
    a=news.loc[news["ID"] == S_norep[i]]
    news2=pandas.concat([news2, a], ignore_index=True)

print(news2)
news2.info(null_counts=True)

 
#estrazione del testo
def vattene(html):
    soup = BeautifulSoup(html) # create a new bs4 object from the html data loaded
    # remove all javascript and stylesheet code
    for script in soup(["script", "style"]): 
        script.extract()
    text = soup.get_text()
    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)
    return text
    
with open("testi.csv", "w") as file:
    writer=csv.writer(file)
    for i in tqdm.tqdm(range(0, len(news2))):
        url=news2.URL[i]
        html = urllib.request.urlopen(url)
        testo=vattene(html)
        writer.writerow([news2.ID[i], testo])
        


#apertura file testi
csv_file = open("testi.csv", encoding="Latin1")
read = pandas.read_csv(csv_file, sep=",", header=None, names=["ID", "Testo"])
print(read.head())

read.info()


############################PREPROCESSING DEI TESTI

#estraiamo le parole dai testi: le salviamo in "parole", una lista di lista di liste di stringhe
type(read.Testo)
parole=[]
for t in range(0,len(read)):
    parole.append(read.Testo[t].split(" "))
   
    
#RIMOZIONE DELLE STOPWORDS   
from nltk.corpus import stopwords
import re
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


#ragiono solo su parole[0]
i=0

#tutto in minuscolo
minuscolo=[]
for j in range(0,len(parole[i])):
    minuscolo.append(parole[i][j].lower())

#rimuoviamo le stopwords
words_nostop=[word for word in minuscolo if word not in stop_words]

#rimuoviamo la punteggiatura
words_nopunct= [word for word in words_nostop if word.isalnum()] 

#part of speach tagging
tagged_words=nltk.pos_tag(words_nopunct) 
togli=eliminare(tagged_words)
indici=np.zeros(int(sum(togli))) 
togli= np.array(togli, dtype=int)
finali=list(np.array(words_nopunct)[togli==0])


csv_file.close()
        