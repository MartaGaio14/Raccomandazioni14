#caso 1
#training set e test set sono costruiti sulle History del test set MIND

# FILE SUL COMPORTAMENTO DEGLI UTENTI: behaviors_test

# apertura del file
import pandas
import tqdm

Set = open("behaviors_test.tsv")
Set1 = pandas.read_csv(Set, sep="\t", header=None, names=["IID", "UID", "Time", "History", "Imp"], usecols=[1, 3])
Set.close()
# pulizia del dataset
GrandeSet = Set1.dropna()
GrandeSet = GrandeSet.reset_index(drop=True)
GrandeSet.info(null_counts=True)
l = []
for i in tqdm.tqdm(range(len(GrandeSet))):
    a = GrandeSet.History[i].split(" ")
    if len(a) > 100:
        l.append(i)
Set_piu100 = GrandeSet.loc[l]  # sono rimasti 229 121 utenti (behaviours ridotto)
Set = Set_piu100.drop_duplicates() #sono rimasti 19181 utenti
Set = Set.reset_index(drop=True)

# campionamento casuale di 1000 utenti (non ripetuti) tra quelli con History maggiori di 100
import numpy as np

np.random.seed(122020)
righe = 1000  # numero di utenti da campionare
campione = np.random.randint(0, len(Set), righe)
dati_camp = Set.loc[campione]
dati_camp = dati_camp.reset_index(drop=True)

Hist = []  # lista di liste news per utente
Id_utente = []  # lista id utenti
for i in range(len(dati_camp)):
    a = dati_camp.History[i].split(" ")
    Hist.append(a)
    Id_utente.append(dati_camp.UID[i])

# eliminiamo le news problematiche ( qualora facessero parte del campione )
# vedi file controllo_url.py
# le news che sono risultate prive di URL sono "N113363", "N110434", "N102010", "N45635"
# le news che compaiono in behaviours.tsv ma non in news.tsv sono "N89741", "N1850"

news_sbagliate = ["N113363", "N110434", "N102010", "N45635", "N89741", "N1850"]
for i in range(len(Hist)):
    for notizia in news_sbagliate:
        if Hist[i].count(notizia) > 0:
            Hist[i].remove(notizia)


# creiamo il corpus completo delle news lette dai 1000 utenti campionati (senza ripetizioni)
tutteNID = []
for i in range(len(Hist)):
    tutteNID += Hist[i]
tutteNID = list(dict.fromkeys(tutteNID))

# FILE CON LE INFORMAZIONI SULLE NEWS: news_test

# apertura del file
news_file = open("news_test.tsv", encoding="Latin1")
read_news = pandas.read_csv(news_file, sep="\t", header=None,
                            names=["ID", "Categoria", "SubCategoria", "Titolo", "Abstract", "URL", "TE", "AE"],
                            usecols=[0, 5])
news_file.close()
read_news = read_news.dropna()
read_news.info(null_counts=True)
read_news = read_news.reset_index(drop=True)

# dal dataset completo vengono selezionate solo le righe contenenti le news che sono in tutteNID
a = []  # lista degli indici delle righe del dataframe da tenere
for i in tqdm.tqdm(range(len(read_news.ID))):
    if read_news.ID[i] in tutteNID:
        a.append(i)
news = read_news.loc[a]
news = news.reset_index(drop=True)

URLS = list(news.URL)

# estrazione del testi
import csv
from preprocessing import extraction

with open("testi.csv", "w", encoding="Utf-8") as file:
    writer = csv.writer(file, delimiter="\t")
    for i in tqdm.tqdm(range(len(URLS))):
        writer.writerow([news.ID[i], extraction(URLS[i])])

# DA TERMINALE: PREPROCESSING CON MAP REDUCE
# python3 MapReduce.py testi.csv > testi_proc.csv
# il file testi_proc conterrà i testi preprocessati


# apertura file testi preprocessati
testi_proc = pandas.read_csv("testi_proc.csv", names=["ID", "parole"], header=None, error_bad_lines=False, sep="\t")

# rimuoviamo da testi_proc le news eliminate post-estrazione
IDvideo = []  # lista con id delle news sbagliate
posvideo = []  # lista con posizioni delle news sbagliate
for i in tqdm.tqdm(range(len(testi_proc.parole))):
    if testi_proc.parole[i] == '[0]':
        IDvideo.append(testi_proc.ID[i])
        posvideo.append(i)
testi_proc = testi_proc.drop(posvideo)
testi_proc = testi_proc.reset_index(drop=True)

# rimuovere anche da Hist gli ID delle news eliminate post-estrazione
for storia in tqdm.tqdm(Hist):
    rem = []
    for codice in storia:
        if codice in IDvideo:
            rem.append(codice)
    for x in rem:
        storia.remove(x)

#divisione delle singole parole preprocessate che vengono lete dal file testi_proc come un'unica stringa
import re
parole = []  # lista di listedelle parole preprocessate per ogni testo
for i in range(len(testi_proc)):
    a = re.sub(r"([^a-zA-Z\s])", "", testi_proc.parole[i])
    parole.append(a.split(" "))


# divisione in training set e test set del corpus delle news
# (viene mantenuta la divisione delle History rispetto ad ogni utente)
n_test = []
n_train = []
for i in range(len(Hist)):
    a = int(len(Hist[i]) * 0.8)
    temp_train = []
    temp_test = []
    for j in range(len(Hist[i])):
        if j < a:
            temp_train.append(Hist[i][j])
        else:
            temp_test.append(Hist[i][j])
    n_train.append(temp_train)
    n_test.append(temp_test)

# lista di tutte le news del training set che sono state lette dalla totalità degli utenti campionati
Storie_train = []
for i in range(len(n_train)):
    Storie_train.extend(n_train[i])
S_norep = list(dict.fromkeys(Storie_train))

Storie_test = []
for i in range(len(n_test)):
    Storie_test.extend(n_test[i])
S_norep2 = list(dict.fromkeys(Storie_test))

# divisione dei testi processati in training e test
testi_train = []
ID_train = []
testi_test = []
ID_test = []
for i in tqdm.tqdm(range(len(testi_proc.ID))):
    if testi_proc.ID[i] in S_norep: #se sta nella lista delle news del training set
        testi_train.append(parole[i])
        ID_train.append(testi_proc.ID[i])
    if testi_proc.ID[i] in S_norep2: #se sta nella lista delle news del test set
        testi_test.append(parole[i])
        ID_test.append(testi_proc.ID[i])


# Rappresentazione in LDA delle news
from gensim import models
import pickle
from profili_item import LDA_corpus

corpus_train, dictionary = LDA_corpus(testi_train)  # creazione del corpus
##alleniamo il modello e lo salviamo su file
# ldamodel = models.LdaMulticore(corpus_train, num_topics=80, id2word=dictionary, passes=20, workers=3)
# pickle.dump(ldamodel, open('lda_model.sav', 'wb'))

# carichiamo il file col modello allenato
ldamodel = pickle.load(open('lda_model.sav', 'rb'))
# rappresentazione in dimensioni latenti di tutti i testi del corpus di train
lda_train = ldamodel[corpus_train]  # lista di liste

# mostra topic e parole associate
from pprint import pprint
pprint(ldamodel.print_topics())
pprint(.get_document_topics(lda_train))

# valutazione del topic model tramite misura di coerenza
coherence_model_lda = models.CoherenceModel(model=ldamodel, texts=testi_train, dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

# visualizzazione dei topic attraverso grafici su internet
import pyLDAvis.gensim
import pyLDAvis
pyLDAvis.enable_notebook()
LDAvis_prepared = pyLDAvis.gensim.prepare(ldamodel, corpus_train, dictionary)
pyLDAvis.show(LDAvis_prepared)


# rappresentazione in dimensioni latenti di tutti i testi del corpus di test sulla base del modello allenato
corpus_test, dictionary = LDA_corpus(testi_test)  # creazione del corpus
lda_test = ldamodel[corpus_test]  # lista di liste

# lista di dizionari, uno per ciascun articolo tra quelli da raccomandare
# rappresentazione utile per il calcolo della similarità
lda_dict_test = []
for i in tqdm.tqdm(range(len(lda_test))):
    lda_dict_test.append(dict(lda_test[i]))

# rappresentazione in TFIDF per le news di training
from profili_item import TFIDF, IDF

idf_train = IDF(testi_train, testi_test)
tfidf_train = TFIDF(testi_train, idf_train)

# rappresentazione in TFIDF per le news di test
# (idf calcolato su dataset di training)
tfidf_test = TFIDF(testi_test, idf_train)

#lista di dizionari: rappresentazione utile per il calcolo della similarità coseno
tfidf_dict_test = [] #lista di dizionari: rappresentazione utile per il calcolo della similarità coseno
for i in tqdm.tqdm(range(len(tfidf_test))):
    tfidf_dict_test.append(dict(tfidf_test[i]))

# CONTENT BASED PROFILE
from profili_utenti import ContentBasedProfile

#creazione dizionari ID : lista di tuple che servono poi per ContentBasedProfile
# profili utenti in rappresentazione lda
diz_lda_train = {}
for i in tqdm.tqdm(range(len(ID_train))):
    diz_lda_train[ID_train[i]] = lda_train[i]

profili_lda = []
for storia in tqdm.tqdm(n_train):
    profili_lda.append(ContentBasedProfile(storia, diz_lda_train))

# profili utenti in rappresentazione tfidf
diz_tfidf_train = {}
for i in tqdm.tqdm(range(len(ID_train))):
    diz_tfidf_train[ID_train[i]] = tfidf_train[i]

profili_tfidf = []
for storia in tqdm.tqdm(n_train):
    profili_tfidf.append(ContentBasedProfile(storia, diz_tfidf_train))


# RACCOMANDAZIONI
from similarita import cosSim
from functools import partial
import multiprocessing as mp

#creazione file che contiene, per ogni combinazione di utente e ID news del test set (=le news da raccomandare),
# la cosine similarity tra il profilo utente e il profilo dell'item, costruiti in entrambe le rappresentazioni (TFIDF/LDA)
N_CPU = mp.cpu_count()
with open("risultati.csv", "w") as file:
    writer = csv.writer(file)
    for i in tqdm.tqdm(range(righe)):  # gira sui 1000 utenti
        pool = mp.Pool(processes = N_CPU)
        f = partial(cosSim, profili_tfidf[i])
        s_tfidf = pool.map(f,  tfidf_dict_test)
        pool.close()
        pool.join()
        for j in range(len(lda_dict_test)):  # gira sulle nuove news
            u = Id_utente[i]
            n = ID_test[j]
            s_lda = cosSim(profili_lda[i], lda_dict_test[j])
            writer.writerow([u, n, s_lda, s_tfidf[j]])


risultati = pandas.read_csv("risultati.csv", names=["UID", "NID", "lda", "tfidf"], header=None, error_bad_lines=False)

# valutazione:  PRECISION-RECALL CURVE
from raccomandazioni import confusion_matrix_par

#calcolo di precisione e richiamo per diverse soglie N
N_grid = list(range(10, len(ID_test), 10))

#LDA
matrici_lda=[]
for N in tqdm.tqdm(N_grid):
    matrici_lda.append(confusion_matrix_par(n_test,"lda", N, ID_test, risultati))

precisioni_lda=[]
richiami_lda=[]
for i in tqdm.tqdm(range(len(N_grid))):
    t=list(zip(*matrici_lda[i]))
    precisioni_lda.append(sum(t[0])/1000)
    richiami_lda.append(sum(t[1])/1000)

import matplotlib.pyplot as plt
plt.plot(richiami_lda, precisioni_lda)
plt.xlabel('richiamo')
plt.ylabel('precisione')
plt.suptitle("LDA")
plt.show()

from sklearn import metrics
auc_lda=metrics.auc(richiami_lda, precisioni_lda)

#TFIDF
matrici_tfidf=[]
for N in tqdm.tqdm(N_grid):
    matrici_tfidf.append(confusion_matrix_par(n_test,"tfidf", N, ID_test, risultati))

precisioni_tfidf=[]
richiami_tfidf=[]
for i in tqdm.tqdm(range(len(N_grid))):
    t=list(zip(*matrici_tfidf[i]))
    precisioni_tfidf.append(sum(t[0])/1000)
    richiami_tfidf.append(sum(t[1])/1000)

plt.plot(richiami_tfidf, precisioni_tfidf)
plt.xlabel('richiamo')
plt.ylabel('precisione')
plt.suptitle("TFIDF")
plt.show()

auc_tfidf=metrics.auc(richiami_tfidf, precisioni_tfidf)