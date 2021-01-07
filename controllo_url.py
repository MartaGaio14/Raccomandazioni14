# CONTROLLO FUNZIONAMENTO URL (per le news tra le quali possiamo campionare - utenti non ripetuti con History di almeno 100  )
#         non è necessario lanciarlo ogni volta

# i dataset
import pandas
import tqdm

Set = open("behaviors_test.tsv")
Set1 = pandas.read_csv(Set, sep="\t", header=None, names=["IID", "UID", "Time", "History", "Imp"], usecols=[1, 3])
Set.close()
GrandeSet = Set1.dropna()
GrandeSet = GrandeSet.reset_index(drop=True)
l = []
for i in tqdm.tqdm(range(len(GrandeSet))):
    a = GrandeSet.History[i].split(" ")
    if len(a) > 100:
        l.append(i)
Set_piu100 = GrandeSet.loc[l]
Set = Set_piu100.drop_duplicates()
Set = Set.reset_index(drop=True)
news_file = open("news_test.tsv", encoding="Latin1")
read_news = pandas.read_csv(news_file, sep="\t", header=None,
                            names=["ID", "Categoria", "SubCategoria", "Titolo", "Abstract", "URL", "TE", "AE"],
                            usecols=[0, 5])
news_file.close()
read_news = read_news.dropna()



Hist_tot = []  # lista di liste news per utente
Storie_tot = []  # lista di tutte le news lette
for i in tqdm.tqdm(range(len(Set))):
    a = Set.History[i].split(" ")
    Hist_tot.append(a)
    Storie_tot.extend(Hist_tot[i])
S_norep = list(dict.fromkeys(Storie_tot))  # devono rimanere 58990 articoli (news ridotto)

read_news = read_news.sort_values("ID")  # le news sono ordinate per codice ID-->per applicare binary search
read_news = read_news.reset_index(drop=True)


def binary_search(data, target, low, high):
    if low > high:
        return "no"
    else:
        mid = (low + high) // 2
        if target == data[mid]:
            return mid
        elif target < data[mid]:
            return binary_search(data, target, low, mid - 1)
        else:
            return binary_search(data, target, mid + 1, high)


pos = []
for code in tqdm.tqdm(S_norep):
    pos.append(binary_search(list(read_news["ID"]), code, 0, len(read_news) - 1))

# controllo parole che sono state lette (in "behaviours") ma che non ci sono in "news"
sbagliato = []
for i in range(len(pos)):
    if pos[i] == "no":
        sbagliato.append(S_norep[i])
        pos.remove("no")
print(sbagliato)
# news "N89741" e "N1850" vanno rimosse dalle History!

# creazione del nuovo dataset che contiene solo le news lette da utenti con History lunghe
news = read_news.loc[pos]
news = news.reset_index(drop=True)

import re
URLS = list(news.URL)
u = re.compile("https")
sbagliato = []
for i in range(len(URLS)):
    if u.match(URLS[i]) is None:
        sbagliato.append(i)
print(news.loc[sbagliato])  # news prive di URL--> "N113363", "N110434", "N102010", "N45635"
