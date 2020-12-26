righe = 1000  # numero di utenti da campionare
Ntfidf = 500  # lunghezza massima della rappresentazione in tfidf
N = 10  # numero di news da raccomandare

##nome file dataset delle news coi body estratti
filename_body = "testi1000.csv"

##nome file modello lda (da scrivere)
filename_lda = 'lda_model_snow1000.sav'

##nome file risultati similarità (da scrivere)
filename_sim = "risultati1000.csv"


###########FILE SUL COMPORTAMENTO DEGLI UTENTI: behaviors_test######################

######## apertura del file
import pandas
import tqdm
Set = open("behaviors_test.tsv")
Set1 = pandas.read_csv(Set, sep="\t", header=None, names=["IID", "UID", "Time", "History", "Imp"], usecols=[1, 3])
Set.close()
#pulizia del dataset
GrandeSet = Set1.dropna()
GrandeSet = GrandeSet.reset_index(drop=True)
GrandeSet.info(null_counts=True)
l = []
for i in tqdm.tqdm(range(len(GrandeSet))):
    a = GrandeSet.History[i].split(" ")
    if len(a) > 150:
        l.append(i)
Set_piu150 = GrandeSet.loc[l] # sono rimasti 229 121 utenti (behaviours ridotto) // 92 970 utenti se len=150
Set_piu150 = Set_piu150.reset_index(drop=True)

#########################FILE CON LE INFORMAZIONI SULLE NEWS: news_test#########################

######## apertura del file
news_file = open("news_test.tsv", encoding="Latin1")
read_news = pandas.read_csv(news_file, sep="\t", header=None,
                            names=["ID", "Categoria", "SubCategoria", "Titolo", "Abstract", "URL", "TE", "AE"],
                            usecols=[0, 5])
news_file.close()
read_news = read_news.dropna()
read_news.info(null_counts=True)

#campionamento casuale di 1000 utenti tra quelli con History maggiori di 100
import numpy as np
np.random.seed(122020)
campione=np.random.randint(0, len(Set_piu150), righe)
dati_camp= Set_piu150.loc[campione]
dati_camp = dati_camp.reset_index(drop=True)
dati_camp.head()

Hist=[] #lista di liste news per utente
Id_utente=[] #lista id utenti
for i in range(len(dati_camp)):
    a=dati_camp.History[i].split(" ")
    Hist.append(a)
    Id_utente.append(dati_camp.UID[i])

###### vedi file controllo_url.py
#le news che sono risultate prive di URL sono "N113363", "N110434", "N102010", "N45635"
#le news che non compaiono in behaviours.tsv ma non in news.tsv sono "N89741", "N1850"

######## eliminiamo le news poblematiche ( qualora facessero parte del campione )
for i in range(len(Hist)):
    if Hist[i].count("N113363") > 0:
        Hist[i].remove("N113363")
    if Hist[i].count("N110434") > 0:
        Hist[i].remove("N110434")
    if Hist[i].count("N102010") > 0:
        Hist[i].remove("N102010")
    if Hist[i].count("N45635") > 0:
        Hist[i].remove("N45635")
    if Hist[i].count("N89741") > 0:
        Hist[i].remove("N89741")
    if Hist[i].count("N1850") > 0:
        Hist[i].remove("N1850")

######## divisione in training set e test set
# (viene mantenuta la divisione delle History rispetto ad ogni utente)
n_test = []
n_training = []
for i in range(len(Hist)):
    a = int(len(Hist[i]) * 0.8)
    temp_train = []
    temp_test = []
    for j in range(len(Hist[i])):
        if j < a:
            temp_train.append(Hist[i][j])
        else:
            temp_test.append(Hist[i][j])
    n_training.append(temp_train)
    n_test.append(temp_test)

##lista di tutte le news del training set che sono state lette dalla totalità degli utenti campionati
Storie_train = []
for i in range(len(n_training)):
    Storie_train.extend(n_training[i])
S_norep = list(dict.fromkeys(Storie_train))

##lista di tutte le news del test set che sono state lette dalla totalità degli utenti campionati
Storie_test = []
for i in range(len(n_test)):
    Storie_test.extend(n_test[i])
S_norep2 = list(dict.fromkeys(Storie_test))

##lista delle news totali per le quali fare il web scraping e il preprocessing
tutteNID = S_norep + S_norep2

# dal dataset completo vengono selezionate solo le righe contenenti le news che sono in tutteNID
news = {}
news = pandas.DataFrame(news)
for i in tqdm.tqdm(range(0, len(tutteNID))):
    a = read_news.loc[read_news["ID"] == tutteNID[i]]
    news = pandas.concat([news, a], ignore_index=True)  # nuovo dataset

URLS = list(news.URL)
# with open("url_file.txt", 'w') as f:
#    for url in URLS:
#        f.write("%s\n" % url)


##estrazione del testo
import requests
from bs4 import BeautifulSoup
import csv
import time

def extraction(url):
    r = requests.get(url, timeout=10)
    if r.status_code == 200:
        soup = BeautifulSoup(r.text, 'html.parser')
        sec =soup.find_all('section')
        if len(sec) > 2:
            body_text = sec[2].text.strip()
        else:
            slides = soup.find_all("div", class_="gallery-caption-text")
            body_text = ""
            for i in range(len(slides)):
                body_text += (slides[i].text.strip())
    return body_text

inizio = time.time()
with open(filename_body, "w", encoding="Utf-8") as file:
    writer = csv.writer(file)
    for i in tqdm.tqdm(range(len(URLS))):
        writer.writerow([news.ID[i], extraction(URLS[i])])
fine = time.time()
print(fine - inizio)

######## apertura file testi
prova = pandas.read_csv(filename_body, names=["ID", "Testo"], header=None, error_bad_lines=False)

# inizio = time.time()
# N_CPU = mp.cpu_count()
# pool = mp.Pool(processes=N_CPU)
# testi_web = pool.map(extraction, URLS[0:1000])
# pool.close()
# pool.join()
# fine = time.time()


print("Fatto web-scraping")

####### preprocessing per tutti i testi

inizio = time.time()
N_CPU = mp.cpu_count()
pool = mp.Pool(processes=N_CPU)
texts = pool.map(preprocessing, list(testi.Testo))
pool.close()
pool.join()
fine = time.time()
print(fine - inizio)

print("Fatto preprocessing")

####### divisione dei testi processati in training e test
# i primi len(S_norep) articoli sono del dataset di training
texts_train = texts[0:len(S_norep) - 1]
texts_test = texts[len(S_norep):len(texts) - 1]

######## Rappresentazione in LDA per le news di training

##creazione del corpus
# frequenza di ogni parola
frequency = defaultdict(int)
for text in texts_train:
    for token in text:
        frequency[token] += 1
# teniamo solo le parole che si ripetono più di una volta
processed_corpus = [[token for token in text if frequency[token] > 1]
                    for text in texts_train]
# a ogni parola associamo un numero
dictionary = corpora.Dictionary(processed_corpus)
# a ogni numero corrispondente alle parole si associa la frequenza
corpus = [dictionary.doc2bow(text) for text in processed_corpus]

##alleniamo il modello e lo salviamo su file... PARTE DA NON FAR GIRARE

ldamodel = models.LdaMulticore(corpus, num_topics=100, id2word=dictionary, passes=20, workers=4)
pickle.dump(ldamodel, open(filename_lda, 'wb'))  # per salvare il modello su file

print("Allenata l'LDA")
# carichiamo il file col modello allenato
# ldamodel = pickle.load(open(filename_lda, 'rb'))

##appresentazione in dimesioni latenti di tutti i testi del corpus

doc_lda = ldamodel[corpus]  # lista di liste
lda_dict = []  # lista di dizionari (utile per risultati)
for i in tqdm.tqdm(range(len(doc_lda))):
    lda_dict.append(dict(doc_lda[i]))

######## Rappresentazione in TFIDF per le news di training

doc_tfidf = TF_IDF(texts_train)

######## Content based profile

# in rappresentazione lda
u_profile_lda = utenti_lda(Hist, doc_lda, S_norep)
print("Calcolati profili utenti LDA")
# in rappresentazione tfidf
u_profile_tfidf = utenti_tfidf_par(Hist, doc_tfidf, S_norep)
print("Calcolati profili utenti TFIDF")

############################# Lavoriamo sul DATASET DI TEST#####################

######## Rappresentazione in LDA per le news di test

##creazione del corpus
# frequenza di ogni parola
frequency2 = defaultdict(int)
for text in texts_test:
    for token in text:
        frequency2[token] += 1
# teniamo solo le parole che si ripetono più di una volta
processed_corpus2 = [[token for token in text if frequency2[token] > 1]
                     for text in texts_test]
# a ogni parola associamo un numero
dictionary2 = corpora.Dictionary(processed_corpus2)
# a ogni numero corrispondente alle parole si associa la frequenza
corpus2 = [dictionary2.doc2bow(text) for text in processed_corpus2]

##appresentazione in dimesioni latenti di tutti i testi del corpus

doc_lda2 = ldamodel[corpus2]  # lista di liste

lda_dict2 = []  # lista di dizionari
for i in tqdm.tqdm(range(len(doc_lda2))):
    lda_dict2.append(dict(doc_lda2[i]))

######## Rappresentazione in TFIDF per le news di training
doc2_tfidf = TF_IDF(texts_test)

print("Calcolate lda e tfidf news di test")
############################# RACCOMANDAZIONI#####################################

# crea vettori di pesi per utenti e news corrispondenti ad uno stesso termine (chiave)

with open(filename_sim, "w") as file:
    writer = csv.writer(file)
    for i in tqdm.tqdm(range(len(u_profile_lda))):  # gira sui 1000 utenti
        for j in range(len(lda_dict2)):  # gira sulle nuove news
            u = Id_utente[i]
            n = S_norep2[j]
            s_lda = similarità(u_profile_lda[i], lda_dict2[j])
            # s_tfidf=similarità(u_profile_tfidf[i],doc2_tfidf[j])
            writer.writerow([u, n, s_lda])

risultati = pandas.read_csv(filename_sim, names=["UID", "NID", "LDA"], header=None, error_bad_lines=False)

##########PRECISIONE RECALL E FALSE POSITIVE RATE
precision = recall = fp_rate = 0
tpli = []
inizio = 0
fine = len(S_norep2)
for u in tqdm.tqdm(range(len(n_test))):
    ut = risultati[inizio:fine]
    top_lda = ut.sort_values(by=['LDA'], ascending=False)[0:N]
    top_lda = top_lda.reset_index(drop=True)
    tp = 0
    for i in range(len(top_lda)):
        if top_lda.NID[i] in n_test[u]:
            tp += 1  # true positive
    tpli.append(tp)
    fp = N - tp  # false positive
    fn = len(n_test[u]) - tp  # false negative
    tn = len(S_norep2) - len(n_test[u]) - fp  # true negative
    precision += tp / (tp + fp)
    recall += tp / (tp + fn)
    fp_rate += fp / (fp + tn)
    inizio = (u + 1) * len(S_norep2) + 1
    fine = (u + 2) * len(S_norep2)

print("PRECISIONE")
print(precision / righe)



from collections import defaultdict
from gensim import corpora, models
import pickle
import time



# importiamo i moduli da noi creati
from preprocessing import *
from tfidf import *
from profili_utenti import utenti_tfidf_par, utenti_tfidf
from profili_utenti import utenti_lda
from similarita import similarità