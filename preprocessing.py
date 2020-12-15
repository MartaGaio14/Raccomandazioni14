
########PREPROCESSING DEI TESTI
#RIMOZIONE DELLE STOPWORDS   
import nltk
from nltk.corpus import stopwords
#from nltk.stem import PorterStemmer
#from nltk.stem import LancasterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import tqdm
import multiprocessing as mp
import threading
import string

stop_words= set(stopwords.words("english"))
    
lettere = list('abcdefghijklmnopqrstuvwxyz')
numeri = list('0123456789')

for i in range(0,len(lettere)):
    stop_words.add(lettere[i])
for i in range(0,len(numeri)):
    stop_words.add(numeri[i])
stop_words.add("getty")
stop_words.add("slides")
verbi_comuni="ask become begin call come could feel find get give go hear help keep know leave let like live look make may mean might move need play put run say see seem show start take talk tell think try turn use want work would said got made went gone knew known took token saw seen came thought gave givenfound told felt left"
verbi_comuni=verbi_comuni.split(" ")

for i in range(len(verbi_comuni)):
    stop_words.add(verbi_comuni[i])
    
#PART OF SPEACH TAGGING
REM=['CC','CD','DT','EX','IN','LS','MD','PDT','POS','PRP','PSRP$','RB',
     'RBR','RBS','TO','UH','WDT','WP','WPD$','WRB', 'RP']

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

#STEMMING
#ps = PorterStemmer() 
#ps=LancasterStemmer()

##LEMMING:
lem = WordNetLemmatizer()

def preprocessing(testi_file):
    #testi_file=testi_train
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
import time
inizio=time.time()
p=preprocessing(testi_train)
fine=time.time()
print(fine-inizio)

#un_testo=testi_file.Testo[i]
def preprocessing2(un_testo):
    #testi_file=testi_train
    #un_testo=testi_train.Testo[0]
    parole=un_testo.split(" ")
    minuscolo=[]
    for j in range(0,len(parole)):
        minuscolo.append(parole[j].lower())
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
    return finali

 #MULTITHREADING 
##ATTENTA: METTI LA CODA in preprocessing2 per fare multithreading    
# def preprocessing_par(tutti_testi):
#     coda = mp.Queue()
#     threads = [mp.Process(target=preprocessing2, args=(un_testo,), 
#         kwargs={"coda": coda}) 
#                for un_testo in tqdm.tqdm(list(tutti_testi.Testo))]   
#     for t in threads:
#         t.start()
#     texts=[coda.get(block=False) for t in hey]
#     for t in threads:
#         t.join() # blocca il MainThread finché t non è completato
#     return texts
#t=preprocessing_par(testi_train)


#MULTIPROCESSING
marta = list(testi_train.Testo[0:10])

N_CPU = mp.cpu_count()
pool=mp.Pool(processes=N_CPU)
texts=pool.map(preprocessing2, marta)
pool.close()
pool.join()











###WORDCLOUD per controllare visivamente il preprocessing
    
def plot_cloud(wordcloud):  
    # Set figure size
    plt.figure(figsize=(40, 30))
    # Display image
    plt.imshow(wordcloud) 
    # No axis details
    plt.axis("off");

# Generate word cloud
def disegna(testo):
    stringa_text=str(testo)
    out = stringa_text.replace("'","")
    out1=out.replace(",","")
    out2=out1.replace("]","")
    out3=out2.replace("[","")
    wordcloud = WordCloud(width = 3000, height = 2000, random_state=1, 
                      collocations=False).generate(out3)
    # Plot
    plot_cloud(wordcloud)
