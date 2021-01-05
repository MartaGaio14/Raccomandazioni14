######## ESTRAZIONE e PREPROCESSING DEI TESTI
import requests
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

url="https://assets.msn.com/labs/mind/AAJ0BJo.html"
def extraction(url):
    r = requests.get(url, timeout=10)
    if r.status_code == 200:  # se l'accesso è avvenuto correttamente ed entro 10 secondi
        soup = BeautifulSoup(r.text, 'html.parser')
        # i diversi tipi di news vengono suddivisi in base al numero di tag <section> che presentano
        sec = soup.find_all("section")
        if len(sec) == 3:  # formato: immagine + solo testo
            a = sec[2].text.strip()  # estrazione del testo
            body_text = re.sub(r"(\n+|\s+)", " ", a)  # salvataggio del testo privato di newline e spazi multipli
            if body_text == '':  # sottocategorie di articoli senza testo ma solo con grafici
                body_text = "sbagliata"
        elif len(sec) == 2:  # formato: varie immagini con porzioni di testo sotto ognuna
            slides = soup.find_all("div",
                                   class_="gallery-caption-text")  # ricerca caption sotto alle immagini (nel tag <div>)
            a = slides[0].text.strip()  # estrazione testo corrispondente alla prima immagine
            # se sotto la prima immagine non compare alcuna caption, viene preso il titolo di ogni immagine
            body_text = ""
            if a != '':  # porzioni di testo trattate come caption
                for i in range(len(slides)):
                    b = slides[i].text.strip()
                    prova = re.sub(r"(\n+|\s+)", " ", b)
                    body_text += " " + prova  # testi di ogni immagine concatenati e separati da spazio
            else:  # porzioni di testo trattate come titoli
                slides = soup.find_all("h2", class_="gallery-title-text")
                for i in range(len(slides)):
                    b = slides[i].text.strip()
                    prova = re.sub(r"(\n+|\s+)", " ", b)
                    body_text += " " + prova
        else:  # formato: video + breve caption
            body_text = "sbagliata"
    return body_text


# RIMOZIONE DELLE STOPWORDS
stop_words = set(stopwords.words("english"))
# aggiunta elementi alla lista predefinita di stopwords
lettere = list('abcdefghijklmnopqrstuvwxyz')
numeri = list('0123456789')
for i in range(0, len(lettere)):
    stop_words.add(lettere[i])
for i in range(0, len(numeri)):
    stop_words.add(numeri[i])
stop_words.add("getty")
stop_words.add("slides")
verbi_comuni = "ask become begin call come could find get give go hear keep know leave let like live look make may " \
               "might move need play put run say see seem show start take tell think try use want work would said got " \
               "made went gone knew known took token saw seen came thought gave given found told left "
verbi_comuni = verbi_comuni.split(" ")
for i in range(len(verbi_comuni)):
    stop_words.add(verbi_comuni[i])


# PART OF SPEACH TAGGING
# rimozione delle parole data la loro funzione grammaticale
def part_of_speach_tagging(words):
    REM = ['CC', 'CD', 'DT', 'EX', 'IN', 'LS', 'MD', 'PDT', 'POS', 'PRP', 'PSRP$', 'RB',
           'RBR', 'RBS', 'TO', 'UH', 'WDT', 'WP', 'WPD$', 'WRB', 'RP'] # tag corrispondenti a parole da rimuovere
    tagged_words = nltk.pos_tag(words)
    tag = list(zip(*tagged_words))  # trasformiamo la lista di tuple (parola-POS) in due tuple [(parole), (POS)]
    w = list(tag[0])
    pos = tag[1]
    da_togliere = []
    for i in range(len(tagged_words)):  # il ciclo prende nota di quali sono le parole da eliminare
        for j in REM:
            if pos[i] == j:
                da_togliere.append(w[i])
    pulite = [word for word in w if word not in da_togliere]
    return pulite


# STEMMING
stemmer = PorterStemmer()


def preprocessing1(un_testo):
    # rimozione della punteggiatura pre-split
    # (spesso punteggiatura usata per divdere le parole, senza ulteriori spazi)
    un_testo = re.sub(r"(\.|,|:|;|!|\?)", " ", un_testo)
    parole = re.split(r"\s+", un_testo)
    minuscolo = []
    for j in range(0, len(parole)):
        minuscolo.append(parole[j].lower())
    # rimozione delle stopwords
    words_nostop = [word for word in minuscolo if word not in stop_words and word.isalpha()]
    # part of speach tagging
    words_post_POS = part_of_speach_tagging(words_nostop)
    # stemming
    stemmed_words = []
    for words in words_post_POS:
        stemmed_words.append(stemmer.stem(words))
    return stemmed_words
