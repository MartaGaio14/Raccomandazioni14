righe = 1000  # numero di utenti da campionare
N = 10  # numero di news da raccomandare

###########FILE SUL COMPORTAMENTO DEGLI UTENTI: behaviors_test######################

######## apertura del file
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
Set_piu100 = GrandeSet.loc[l]  # sono rimasti 229 121 utenti (behaviours ridotto) // 92 970 utenti se len=150
Set_piu100 = Set_piu100.reset_index(drop=True)

# campionamento casuale di 1000 utenti tra quelli con History maggiori di 100
import numpy as np

np.random.seed(122020)
campione = np.random.randint(0, len(Set_piu100), righe)
dati_camp = Set_piu100.loc[campione]
dati_camp = dati_camp.reset_index(drop=True)

Hist = []  # lista di liste news per utente
Id_utente = []  # lista id utenti
for i in range(len(dati_camp)):
    a = dati_camp.History[i].split(" ")
    Hist.append(a)
    Id_utente.append(dati_camp.UID[i])

######## eliminiamo le news poblematiche ( qualora facessero parte del campione )
###### vedi file controllo_url.py
# le news che sono risultate prive di URL sono "N113363", "N110434", "N102010", "N45635"
# le news che compaiono in behaviours.tsv ma non in news.tsv sono "N89741", "N1850"

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

##creiamo il corpus completo delle news lette dai 1000 utenti campionati (senza ripetizioni)
tutteNID = []
for i in range(len(Hist)):
    tutteNID += Hist[i]
tutteNID = list(dict.fromkeys(tutteNID))

#########################FILE CON LE INFORMAZIONI SULLE NEWS: news_test#########################

######## apertura del file
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

##estrazione del testi
import csv
from preprocessing import extraction, preprocessing1

with open("testi.csv", "w", encoding="Utf-8") as file:
    writer = csv.writer(file, delimiter="\t")
    for i in tqdm.tqdm(range(len(URLS))):
        writer.writerow([news.ID[i], extraction(URLS[i])])

# preprocessing calcolo parallelo
# import multiprocessing as mp
# import time
#
# testiweb=pandas.read_csv("testi.csv", names=["ID", "testo"], header=None, error_bad_lines=False, sep="\t")
# inizio=time.time()
# N_CPU = mp.cpu_count()
# pool=mp.Pool(processes=N_CPU)
# texts2=pool.map(preprocessing1, list(testiweb.testo))
# pool.close()
# pool.join()
# fine=time.time()
# print(fine-inizio)


#######DA TERMINALE: PREPROCESSING CON MAP REDUCE
# python3 MapReduce.py testi.csv > testi_proc.csv
# testi_proc conterrà i testi preprocessati


######## apertura file testi preprocessati
testi_proc = pandas.read_csv("testi_proc.csv", names=["ID", "parole"], header=None, error_bad_lines=False, sep="\t")

##rimuoviamo da testi_proc e da Hist le news con video
IDvideo = []  # lista con id delle news con video
posvideo = []  # lista con posizioni delle news con video
for i in tqdm.tqdm(range(len(testi_proc.parole))):
    if testi_proc.parole[i] == '[0]':
        IDvideo.append(testi_proc.ID[i])
        posvideo.append(i)

testi_proc = testi_proc.drop(posvideo)
testi_proc = testi_proc.reset_index(drop=True)

import re

parole = []  # lista delle parole preprocessate per ogni testo
for i in range(len(testi_proc)):
    a = re.sub(r"([^a-zA-Z\s])", "", testi_proc.parole[i])
    parole.append(a.split(" "))

# rimuovere anche da Hist gli ID delle news eliminate post-estrazione
for storia in tqdm.tqdm(Hist):
    rem = []
    for codice in storia:
        if codice in IDvideo:
            rem.append(codice)
    for x in rem:
        storia.remove(x)

######## divisione in training set e test set del corpus delle news
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

##lista di tutte le news del training set che sono state lette dalla totalità degli utenti campionati
Storie_train = []
for i in range(len(n_train)):
    Storie_train.extend(n_train[i])
S_norep = list(dict.fromkeys(Storie_train))

##lista di tutte le news del test set che sono state lette dalla totalità degli utenti campionati
Storie_test = []
for i in range(len(n_test)):
    Storie_test.extend(n_test[i])

###### divisione dei testi processati in training e test
testi_train = []
ID_train = []
testi_test = []
ID_test = []
for i in tqdm.tqdm(range(len(testi_proc.ID))):
    if testi_proc.ID[i] in S_norep:
        testi_train.append(parole[i])
        ID_train.append(testi_proc.ID[i])
    else:
        testi_test.append(parole[i])
        ID_test.append(testi_proc.ID[i])


######## Rappresentazione in LDA delle news

##alleniamo il modello e lo salviamo su file... PARTE DA NON FAR GIRARE
from gensim import models
import pickle
from profili_item import LDA_corpus

corpus_train, dictionary = LDA_corpus(testi_train)  # creazione del corpus

#ldamodel = models.LdaMulticore(corpus_train, num_topics=80, id2word=dictionary, passes=20, workers=3)
#pickle.dump(ldamodel, open('lda_model.sav', 'wb'))  # per salvare il modello su file

# carichiamo il file col modello allenato
ldamodel = pickle.load(open('lda_model.sav', 'rb'))
# from pprint import pprint
# pprint(ldamodel.print_topics())

coherence_model_lda = models.CoherenceModel(model=ldamodel, texts=testi_train, dictionary=dictionary, coherence='c_v')
coherence_lda80 = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda80)

import pyLDAvis.gensim
import pyLDAvis
# Visualize the topics (grafici su internet)
pyLDAvis.enable_notebook()
LDAvis_prepared80 = pyLDAvis.gensim.prepare(ldamodel, corpus_train, dictionary)
pyLDAvis.show(LDAvis_prepared80)

# rappresentazione in dimensioni latenti di tutti i testi del corpus di train
lda_train = ldamodel[corpus_train]  # lista di liste


# rappresentazione in dimesioni latenti di tutti i testi del corpus di test sulla base del modello allenato
corpus_test, dictionary = LDA_corpus(testi_test)  # creazione del corpus
lda_test = ldamodel[corpus_test]  # lista di liste


######## Rappresentazione in TFIDF per le news di training
from profili_item import TFIDF, IDF

idf_train = IDF(testi_train, testi_test)
tfidf_train = TFIDF(testi_train, idf_train)

# rappresentazione in TFIDF per le news di test
# (idf calcolato su dataset di training)
tfidf_test = TFIDF(testi_test, idf_train)

######## Content based profile
from profili_utenti import ContentBasedProfile

#creazione dizionari ID : lista di tuple che servono poi per ContentBasedProfile
diz_lda_train = {}
for i in tqdm.tqdm(range(len(ID_train))):
    diz_lda_train[ID_train[i]] = lda_train[i]

diz_tfidf_train = {}
for i in tqdm.tqdm(range(len(ID_train))):
    diz_tfidf_train[ID_train[i]] = tfidf_train[i]

# in rappresentazione lda
profili_lda = []
for storia in tqdm.tqdm(n_train):
    profili_lda.append(ContentBasedProfile(storia, diz_lda_train))

# in rappresentazione tfidf
profili_tfidf = []
for storia in tqdm.tqdm(n_train):
    profili_tfidf.append(ContentBasedProfile(storia, diz_tfidf_train))

############################# RACCOMANDAZIONI#####################################
from similarita import cosSim

lda_dict_test = []  # lista di dizionari, uno per ciascun articolo tra quelli da raccomandare
for i in tqdm.tqdm(range(len(lda_test))):
    lda_dict_test.append(dict(lda_test[i]))

tfidf_dict_test = []
for i in tqdm.tqdm(range(len(tfidf_test))):
    tfidf_dict_test.append(dict(tfidf_test[i]))

# crea vettori di pesi per utenti e news corrispondenti ad uno stesso termine (chiave)
import csv
with open("risultati.csv", "w") as file:
    writer = csv.writer(file)
    for i in tqdm.tqdm(range(righe)):  # gira sui 1000 utenti
        for j in range(len(lda_dict_test)):  # gira sulle nuove news
            u = Id_utente[i]
            n = ID_test[j]
            s_lda = cosSim(profili_lda[i], lda_dict_test[j])
            s_tfidf = cosSim(profili_tfidf[i], tfidf_dict_test[j])
            writer.writerow([u, n, s_lda, s_tfidf])

risultati = pandas.read_csv("risultati.csv", names=["UID", "NID", "LDA"], header=None, error_bad_lines=False)


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
