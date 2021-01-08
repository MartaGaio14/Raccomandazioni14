import multiprocessing as mp
import threading
from sklearn.metrics import ndcg_score

# pos_utente è la posizione dell'utente in ID_utente (da 1 a 1000)
# tipo è il tipo di rappresentazione, "lda" o "tfidf"
# N è il numero di news da raccomandare
# ID_test è lista di news candidate alla raccomandazione
def raccomandati(pos_utente, tipo, N, ID_test, risultati):
    inizio = pos_utente * len(ID_test)
    fine = inizio + len(ID_test)
    ut = risultati[inizio:fine] #selezione dati relativi all'utente in questione
    ut = ut.sort_values(by=[tipo], ascending=False) #ordine decrescente dei dati rispetto alla similarità
    ut = ut.reset_index(drop=True)
    top_tipo = ut[0:N]
    return list(top_tipo.NID)

def raccomandati2(pos_utente, tipo, N, imp, risultati):
    inizio = 0
    if pos_utente != 0:
        for i in range(pos_utente - 1):
            inizio += len(imp[i])
    fine = inizio + len(imp[pos_utente])
    ut = risultati[inizio:fine]  # selezione dati relativi all'utente in questione
    ut = ut.sort_values(by=[tipo], ascending=False)  # ordine decrescente dei dati rispetto alla similarità
    ut = ut.reset_index(drop=True)
    top_tipo = ut[0:N]
    return list(top_tipo.NID)


# storia: lista delle news lette dall'utente (n_test[i])
def confusion_matrix(pos_utente, storia, tipo, N, ID_test, risultati, coda=None):
    racc = raccomandati(pos_utente, tipo, N, ID_test, risultati) # lista delle news raccomandate per l'utente
    racc_set = set(racc)
    intersection = list(racc_set.intersection(storia))  # news lette e raccomandate
    tp = len(intersection)
    fp = N - tp  # false positive
    fn = len(storia) - tp  # false negative
    tn = len(ID_test) - len(storia) - fp  # true negative
    precision = tp / (tp + fp)  # equivalente a tp/ N
    recall = tp / (tp + fn) # equivalente a tp / len(storia)
    if coda is None:  # se è non valorizzato ritorno il risultato
        return (precision, recall)
    else:  # altrimenti accodo il risultato che ho trovato
        return coda.put((precision, recall))

#funzione parallelizzata tramite multithreding
def confusion_matrix_par(n_test, tipo, N, ID_test, risultati):
    coda = mp.Queue()
    #args: argomenti dinamici, la posizione dell'utente e la lista delle news del test set che ha letto (storia)
    #kwargs: argomenti statici
    threads = [threading.Thread(target=confusion_matrix, args=(pos_utente, n_test[pos_utente],),
                                kwargs={"tipo": tipo, "N": N, "ID_test": ID_test, "risultati": risultati, "coda": coda})
               for pos_utente in range(len(n_test))]
    for t in threads:
        t.start()
    x = [coda.get() for t in threads]
    for t in threads:
        t.join()  # blocca il MainThread finché t non è completato
    return x

#per ogni utente vengono inseriti imp: il dizionario contenente la sua impression, racc: la lista delle news raccomandate
def input_ndcg(imp, racc):
    true = list(imp.values())
    ID = list(imp.keys())
    prev = []
    for articolo in ID:
        if articolo in racc:
            prev.append(1)
        else:
            prev.append(0)
    return true, prev

#calcola il punteggio ndcg
def ndcg(pos_utente, tipo, N, imp, risultati):
    racc = raccomandati2(pos_utente, tipo,  N, imp, risultati)
    true, prev = input_ndcg(imp[pos_utente], racc)
    return ndcg_score(true, prev)
