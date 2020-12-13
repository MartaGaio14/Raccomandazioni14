
########PREPROCESSING DEI TESTI
#RIMOZIONE DELLE STOPWORDS   
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import tqdm

stop_words= set(stopwords.words("english"))
    
lettere = list('abcdefghijklmnopqrstuvwxyz')
numeri = list('0123456789')


for i in range(0,len(lettere)):
    stop_words.add(lettere[i])
for i in range(0,len(numeri)):
    stop_words.add(numeri[i])
stop_words.add("getty")

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
ps = PorterStemmer() 
#ps=LancasterStemmer()

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
        #stemming
        stemmed_words=[]
        for w in finali:
            stemmed_words.append(ps.stem(w))
        finali=stemmed_words
        #salvo le parole rimaste per ogni documento in una lista di liste
        texts.append(finali)
    return texts

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
