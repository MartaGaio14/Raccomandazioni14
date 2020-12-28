######## PREPROCESSING DEI TESTI
#RIMOZIONE DELLE STOPWORDS   
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np

def vattene(url):
    r = requests.get(url, timeout=10)
    soup = BeautifulSoup(r.text, 'html.parser') # crea oggetto bs4 dal link html
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

def extraction(url):
    r = requests.get(url, timeout=10)
    if r.status_code == 200:
        soup = BeautifulSoup(r.text, 'html.parser')
        sec=soup.find_all("section")
        if len(sec) == 3: ##belle, tipo URLS[0]
            sec=soup.find_all("section")
            body_text = sec[2].text.strip()
        elif len(sec) == 2: ##brutte, tipo URLS[1]
            slides = soup.find_all("div", class_="gallery-caption-text")
            body_text = ""
            for i in range(len(slides)):
                body_text += (slides[i].text.strip())
        else:##tipo il video URLS[182]
            body_text="sbagliata"
    return body_text


stop_words= set(stopwords.words("english"))
lettere = list('abcdefghijklmnopqrstuvwxyz')
numeri = list('0123456789')
for i in range(0,len(lettere)):
    stop_words.add(lettere[i])
for i in range(0,len(numeri)):
    stop_words.add(numeri[i])
stop_words.add("getty")
stop_words.add("slides")
verbi_comuni = "ask become begin call come could find get give go hear keep know leave let like live look make may might move need play put run say see seem show start take tell think try use want work would said got made went gone knew known took token saw seen came thought gave given found told left"
verbi_comuni = verbi_comuni.split(" ")
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

stemmer = PorterStemmer()
lem = WordNetLemmatizer()

def preprocessing1(un_testo):
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
    #stemming
    stemmed_words=[]
    for w in finali:
        stemmed_words.append(stemmer.stem(w))
    finali=stemmed_words
    #lemming
    # lemmed_words=[]
    # for w in finali:
    #     lemmed_words.append(lem.lemmatize(w,"v"))
    # finali=lemmed_words
    return finali

