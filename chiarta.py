#caso 2
# il training set è la History e il test set corrisponde alle Impressions del training set MIND

# FILE SUL COMPORTAMENTO DEGLI UTENTI: behaviors_training

# apertura del file
import pandas
import tqdm

Set = open("behaviors_train.tsv")
Set1 = pandas.read_csv(Set, sep="\t", header=None, names=["IID", "UID", "Time", "History", "Imp"], usecols=[1, 3, 4])
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
Set_piu100 = GrandeSet.loc[l]  # sono rimasti 137 605 utenti
Set_piu100['Imp'] = Set_piu100.groupby(['UID', 'History'])['Imp'].transform(lambda x : ' '.join(x))
Set_piu100 = Set_piu100.drop_duplicates() #sono rimasti 11 576 utenti
Set = Set_piu100.reset_index(drop=True)


# campionamento casuale di 1000 utenti (non ripetuti) tra quelli con History maggiori di 100
import numpy as np
np.random.seed(122020)
righe = 500  # numero di utenti da campionare
campione = np.random.randint(0, len(Set), righe)
dati_camp = Set.loc[campione]
dati_camp = dati_camp.reset_index(drop=True)

Hist = []  # lista di liste news per utente
Id_utente = []  # lista id utenti
Impr = []
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


# creiamo il corpus completo delle news lette e viste dai 500 utenti campionati (senza ripetizioni)
tutteNID = []
for i in range(len(Hist)):
    tutteNID.extend(Hist[i])
    tutteNID.extend(Impr[i].keys())
tutteNID = list(dict.fromkeys(tutteNID))

# FILE CON LE INFORMAZIONI SULLE NEWS: news_test

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
