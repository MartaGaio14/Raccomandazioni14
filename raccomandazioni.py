import multiprocessing as mp
import threading
import tqdm

#pos_utente è la posizione dell'utente in ID_utente
#tipo è "lda" o "tfidf"
def raccomandati(pos_utente, tipo, N, ID_test, risultati):
    inizio = (pos_utente + 1) * len(ID_test) + 1
    fine = (pos_utente + 2) * len(ID_test)
    ut = risultati[inizio:fine]
    top_tipo = ut.sort_values(by=[tipo], ascending=False)[0:N]
    top_tipo = top_tipo.reset_index(drop=True)
    return top_tipo

#racc lista delle news raccomandate per l'utente (output raccomandati (lda o tfidf))
#storia lista delle news lette dall'utente (n_test[i])
def confusion_matrix(pos_utente,storia,tipo, N, ID_test, risultati, coda=None):
    racc=raccomandati(pos_utente,tipo, N, ID_test, risultati)
    racc_set = set(racc)
    intersection = list(racc_set.intersection(storia)) #news lette e raccomandate
    tp=len(intersection)
    fp = N - tp  # false positive
    fn = len(storia) - tp  # false negative
    tn = len(ID_test) - len(storia) - fp  # true negative
    if coda is None:#se è non valorizzato ritorno il risultato
        return (tp, fn, fp, tn)
    else:#altrimenti accodo il risultato che ho trovato
        coda.put((tp, fn, fp, tn))

def confusion_matrix_par(n_test, tipo, N, ID_test, risultati):
    coda = mp.Queue()
    threads = [threading.Thread(target=confusion_matrix, args=(pos_utente,n_test[pos_utente]),
        kwargs={"tipo":tipo, "N":N, "ID_test": ID_test, "risultati": risultati, "coda": coda})
               for pos_utente in tqdm.tqdm(range(len(n_test)))]
    for t in threads:
        t.start()
    x=[coda.get() for t in threads]
    for t in threads:
        t.join() # blocca il MainThread finché t non è completato
    return x