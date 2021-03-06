#caso 2
# il training set è la History e il test set corrisponde alle Impressions del training set MIND
import pandas
import tqdm
import numpy as np
import csv
import re
from gensim import models
import pickle
from functools import partial
import multiprocessing as mp

from preprocessing import extraction
from profili_item import LDA_corpus
from profili_item import TFIDF, IDF
from profili_utenti import ContentBasedProfile
from raccomandazioni import confusion_matrix_par2
from similarita import cosSim

# FILE SUL COMPORTAMENTO DEGLI UTENTI: behaviors_training
# apertura del file
Set = open("behaviors_train.tsv")
Set1 = pandas.read_csv(Set, sep="\t", header=None, names=["IID", "UID", "Time", "History", "Imp"], usecols=[1, 3, 4])
Set.close()
# pulizia del dataset
GrandeSet = Set1.dropna()
GrandeSet = GrandeSet.reset_index(drop=True)

l = [] # utenti con History di lunghezza maggiore di 100
for i in tqdm.tqdm(range(len(GrandeSet))):
    a = GrandeSet.History[i].split(" ")
    if len(a) > 100:
        l.append(i)
Set_piu100 = GrandeSet.loc[l]  # sono rimasti 137 605 utenti
Set_piu100['Imp'] = Set_piu100.groupby(['UID', 'History'])['Imp'].transform(lambda x : ' '.join(x)) # concatenazione
# delle impression di uno stesso utente
Set_piu100 = Set_piu100.drop_duplicates() #sono rimasti 11 576 utenti
Set = Set_piu100.reset_index(drop=True)


# campionamento casuale di 1000 utenti (non ripetuti) tra quelli con History maggiori di 100
np.random.seed(122020)
righe = 1000  # numero di utenti da campionare
campione = np.random.randint(0, len(Set), righe)
dati_camp = Set.loc[campione]
dati_camp = dati_camp.reset_index(drop=True)

Hist = []  # lista di liste di news lette da ogni utente
Id_utente = []  # lista id utenti
Impr = [] # lista di dizionari, ciscun dizionario contiene le impressions di un utente (chiave: id della news, valore:
# 0 o 1 a seconda che quella news sia stata cliccata o meno dall'utente)
for i in range(len(dati_camp)):
    a = dati_camp.History[i].split(" ")
    Hist.append(a)
    b = dati_camp.Imp[i].split(" ")
    diz_temp={}
    for el in b:
        c = el.split("-")
        diz_temp[c[0]] = c[1]
    Impr.append(diz_temp)
    Id_utente.append(dati_camp.UID[i])


# eliminiamo le news problematiche ( qualora facessero parte del campione )
news_sbagliate = ["N113363", "N110434", "N102010", "N45635", "N89741", "N1850"]
for i in range(len(Hist)):
    for notizia in news_sbagliate:
        if Hist[i].count(notizia) > 0:
            Hist[i].remove(notizia)
        if list(Impr[i].keys()).count(notizia) > 0:
            del Impr[i][notizia]


# creiamo il corpus completo delle news lette e viste dai 1000 utenti campionati (senza ripetizioni)
tutteNID = []
for i in range(len(Hist)):
    tutteNID.extend(Hist[i])
    tutteNID.extend(Impr[i].keys())
tutteNID = list(dict.fromkeys(tutteNID))

# FILE CON LE INFORMAZIONI SULLE NEWS: news_test
"""
# apertura del file
news_file = open("news_train.tsv", encoding="Latin1")
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
with open("testi2.csv", "w", encoding="Utf-8") as file:
    writer = csv.writer(file, delimiter="\t")
    for i in tqdm.tqdm(range(len(URLS))):
        writer.writerow([news.ID[i], extraction(URLS[i])])

# DA TERMINALE: PREPROCESSING CON MAP REDUCE
# python MapReduce.py testi2.csv > testi_proc2.csv

"""

# apertura file testi preprocessati
testi_proc = pandas.read_csv("testi_proc2.csv", names=["ID", "parole"], header=None, error_bad_lines=False, sep="\t")

IDvideo = []  # lista con id delle news sbagliate
posvideo = []  # lista con posizioni delle news sbagliate
for i in tqdm.tqdm(range(len(testi_proc.parole))):
    if testi_proc.parole[i] == '[0]':
        IDvideo.append(testi_proc.ID[i])
        posvideo.append(i)
testi_proc = testi_proc.drop(posvideo)
testi_proc = testi_proc.reset_index(drop=True)

# rimuovere anche da Hist e da Impr gli ID delle news eliminate post-estrazione
for i in tqdm.tqdm(range(len(Hist))):
    rem_h = []
    rem_i =[]
    for codice in Hist[i]:
        if codice in IDvideo:
            rem_h.append(codice)
    for x in rem_h:
        Hist[i].remove(x)
    for codice in list(Impr[i].keys()):
        if codice in IDvideo:
            rem_i.append(codice)
    for x in rem_i:
        del Impr[i][x]


# divisione delle singole parole preprocessate che vengono lette dal file testi_proc2 come un'unica stringa
parole = []  # lista di liste delle parole preprocessate per ogni testo
for i in range(len(testi_proc)):
    a = re.sub(r"([^a-zA-Z\s])", "", testi_proc.parole[i])
    parole.append(a.split(" "))


# divisione in training set e test set del corpus delle news
n_train = Hist # le news del training set sono quelle presenti nelle History
n_test =[] # le news del training set sono quelle presenti nelle Impressions
for signore in Impr:
    n_test.append(list(signore.keys()))


# lista di tutte le news del training set che sono state lette dalla totalità degli utenti campionati
Storie_train = []
for i in range(len(n_train)):
    Storie_train.extend(n_train[i])
S_norep = list(dict.fromkeys(Storie_train))

# lista di tutte le news del test set che sono state lette dalla totalità degli utenti campionati
Storie_test = []
for i in range(len(n_test)):
    Storie_test.extend(n_test[i])
S_norep2 = list(dict.fromkeys(Storie_test))


# divisione dei testi processati in training e test
testi_train = []
ID_train = []  # riscriviamo la lista degli id delle news di training nell'ordine in cui esse compaiono in testi_proc2
testi_test = []
ID_test = [] # riscriviamo la lista degli id delle news di test nell'ordine in cui esse compaiono in testi_proc2
# è necessario riscrivere ID_train e ID_test perchè, usando MapReduce nella creazione di testi_proc2, cambia ogni volta
# l'ordine con cui vengono restituiti i testi
for i in tqdm.tqdm(range(len(testi_proc.ID))):
    if testi_proc.ID[i] in S_norep: #se sta nella lista delle news del training set
        testi_train.append(parole[i])
        ID_train.append(testi_proc.ID[i])
    if testi_proc.ID[i] in S_norep2:  # se sta nella lista delle news del test set
        testi_test.append(parole[i])
        ID_test.append(testi_proc.ID[i])


# Rappresentazione in LDA delle news
corpus_train, dictionary = LDA_corpus(testi_train)  # creazione del corpus
"""
#alleniamo il modello e lo salviamo su file
ldamodel = models.LdaMulticore(corpus_train, num_topics=80, id2word=dictionary, passes=20, workers=3)
pickle.dump(ldamodel, open('lda_model2.sav', 'wb'))
"""
# carichiamo il file col modello allenato
ldamodel = pickle.load(open('lda_model2.sav', 'rb'))

# rappresentazione in dimensioni latenti di tutti i testi del corpus di train
lda_train = ldamodel[corpus_train]  # lista di liste


"""
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
"""

# rappresentazione in dimensioni latenti di tutti i testi del corpus di test sulla base del modello allenato
corpus_test, dictionary = LDA_corpus(testi_test)  # creazione del corpus
lda_test = ldamodel[corpus_test]  # lista di liste

lda_test=list(ldamodel.get_document_topics(corpus_test))

# lista di dizionari, uno per ogni utente. Ogni dizionario contiene la rappresentazione in LDA delle news presenti
# nell'impression dell'utente (chiave: ID della news, valore: dizionario con la sua rappresentazione LDA)
lda_dict_test = []
for imp in tqdm.tqdm(n_test): # imp è la lista delle news presenti nell'impression dell'utente
    dt = {}
    for i in range(len(lda_test)):
        if ID_test[i] in imp: # ID_test[i] è l'id della news la cui rappresentazione LDA sta in lda_test[i] (stesso ordine)
            dt[ID_test[i]]=dict(lda_test[i])
    lda_dict_test.append(dt)

# rappresentazione in TFIDF per le news di training
idf_train = IDF(testi_train, testi_test)
tfidf_train = TFIDF(testi_train, idf_train)

# rappresentazione in TFIDF per le news di test
# (idf calcolato su dataset di training)
tfidf_test = TFIDF(testi_test, idf_train)

# lista di dizionari, uno per ogni utente. Ogni dizionario contiene la rappresentazione in TFIDF delle news presenti
# nell'impression dell'utente (chiave: ID della news, valore: dizionario con la sua rappresentazione TFIDF)
tfidf_dict_test = []
for imp in tqdm.tqdm(n_test): # imp è la lista delle news presenti nell'impression dell'utente
    dt = {}
    for i in range(len(ID_test)):
        if ID_test[i] in imp: # ID_test[i] è l'id della news la cui rappresentazione TFIDF sta in lda_test[i] (stesso ordine)
            dt[ID_test[i]]=dict(tfidf_test[i])
    tfidf_dict_test.append(dt)

# CONTENT BASED PROFILE
# profili utenti in rappresentazione lda
diz_lda_train = {} # dizionario chiave: ID della news di training, valore: lista di tuple con la sua rappresentazione lda
# rappresentazione utile per ContentBasedProfile
for i in tqdm.tqdm(range(len(ID_train))):
    diz_lda_train[ID_train[i]] = lda_train[i]
profili_lda = [] # lista di dizionari: ciascun dizionario contiene il profilo lda di un utente
for storia in tqdm.tqdm(n_train):
    profili_lda.append(ContentBasedProfile(storia, diz_lda_train))

# profili utenti in rappresentazione tfidf
diz_tfidf_train = {}# dizionario chiave: ID della news di training, valore: lista di tuple con la sua rappresentazione tfidf
# rappresentazione utile per ContentBasedProfile
for i in tqdm.tqdm(range(len(ID_train))):
    diz_tfidf_train[ID_train[i]] = tfidf_train[i]
profili_tfidf = []  #lista di dizionari: ciascun dizionario contiene il profilo tfidf di un utente
for storia in tqdm.tqdm(n_train):
    profili_tfidf.append(ContentBasedProfile(storia, diz_tfidf_train))

# RACCOMANDAZIONI
"""
#creazione file che contiene, per ogni combinazione di utente e ID news delle sue impressions (=le news da raccomandare),
# la cosine similarity tra il profilo utente e il profilo dell'item, costruiti in entrambe le rappresentazioni (TFIDF/LDA)
N_CPU = mp.cpu_count()
with open("risultati2.csv", "w") as file:
    writer = csv.writer(file)
    for i in tqdm.tqdm(range(righe)):  # gira sui 1000 utenti
        pool = mp.Pool(processes = N_CPU)
        f = partial(cosSim, profili_tfidf[i])
        dr_tfidf=list(tfidf_dict_test[i].values()) #lista degli articoli candidati alla raccomandazione per l'i-esimo utente
        #ciascun articolo è espresso tramite dizionario con la sua rappresentazione in tfidf
        s_tfidf = pool.map(f, dr_tfidf) # lista delle similarità tra utente i e tutte le news del suo test set
        pool.close()
        pool.join()
        drid = list(lda_dict_test[i].keys())  # lista degli id degli articoli candidati alla raccomandazione per
        # l'i-esimo utente
        dr_lda = list(lda_dict_test[i].values())  # lista degli articoli candidati alla raccomandazione per l'i-esimo utente
        # ciascun articolo è espresso tramite dizionario con la sua rappresentazione in lda
        for j in range(len(lda_dict_test[i])):  # gira sulle nuove news
            u = Id_utente[i]
            n = drid[j]
            s_lda = cosSim(profili_lda[i], dr_lda[j]) # similarità tra utente i e news j
            writer.writerow([u, n, s_lda, s_tfidf[j]])
"""

risultati = pandas.read_csv("risultati2.csv", names=["UID", "NID", "lda", "tfidf"], header=None, error_bad_lines=False)

# valutazione: PRECISION-RECALL
N_grid=[5,10,20] # calcoliamo precisione richiamo medi per questi diversi valori di N (numero di news da raccomandare)

#lda
matrici_lda=[]
for N in tqdm.tqdm(N_grid):
    matrici_lda.append(confusion_matrix_par2(n_test,"lda", N, Impr, risultati))

precisioni_lda=[]
richiami_lda=[]
for i in tqdm.tqdm(range(len(N_grid))):
    t=list(zip(*matrici_lda[i]))
    precisioni_lda.append(sum(t[0])/righe)
    richiami_lda.append(sum(t[1])/righe)

#tfidf
matrici_tfidf=[]
for N in tqdm.tqdm(N_grid):
    matrici_tfidf.append(confusion_matrix_par2(n_test,"tfidf", N, Impr, risultati))

precisioni_tfidf=[]
richiami_tfidf=[]
for i in tqdm.tqdm(range(len(N_grid))):
    t=list(zip(*matrici_tfidf[i]))
    precisioni_tfidf.append(sum(t[0])/righe)
    richiami_tfidf.append(sum(t[1])/righe)
