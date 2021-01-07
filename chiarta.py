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
for i in range(len(dati_camp)):
    a = dati_camp.History[i].split(" ")
    Hist.append(a)
    Id_utente.append(dati_camp.UID[i])

# eliminiamo le news problematiche ( qualora facessero parte del campione )
news_sbagliate = ["N113363", "N110434", "N102010", "N45635", "N89741", "N1850"]
for i in range(len(Hist)):
    for notizia in news_sbagliate:
        if Hist[i].count(notizia) > 0:
            Hist[i].remove(notizia)
#controllare anche le news di IMP!!

# creiamo il corpus completo delle news lette dai 1000 utenti campionati (senza ripetizioni)
tutteNID = []
for i in range(len(Hist)):
    tutteNID += Hist[i]
tutteNID = list(dict.fromkeys(tutteNID))