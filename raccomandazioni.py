import multiprocessing as mp
import threading


# pos_utente è la posizione dell'utente in ID_utente (da 1 a
# tipo è il tipo di rappresentazione, "lda" o "tfidf"
def raccomandati(pos_utente, tipo, N, ID_test, risultati):
    inizio = pos_utente * len(ID_test)
    fine = inizio + len(ID_test)
    ut = risultati[inizio:fine]
    ut = ut.sort_values(by=[tipo], ascending=False)
    ut = ut.reset_index(drop=True)
    top_tipo = ut[0:N]
    return list(top_tipo.NID)


# racc: lista delle news raccomandate per l'utente (output raccomandati (lda o tfidf))
# storia: lista delle news lette dall'utente (n_test[i])
def confusion_matrix(pos_utente, storia, tipo, N, ID_test, risultati, coda=None):
    racc = raccomandati(pos_utente, tipo, N, ID_test, risultati)
    racc_set = set(racc)
    intersection = list(racc_set.intersection(storia))  # news lette e raccomandate
    tp = len(intersection)
    fp = N - tp  # false positive
    fn = len(storia) - tp  # false negative
    tn = len(ID_test) - len(storia) - fp  # true negative
    # Precision@k = (# of recommended items @k that are relevant) / (# of recommended items @k)
    # precision_N = tp/ N
    precision = tp / (tp + fp)
    # Recall @ k = (  # of recommended items @k that are relevant) / (total # of relevant items)
    # recall_N =  tp / len(storia)
    recall = tp / (tp + fn)
    if coda is None:  # se è non valorizzato ritorno il risultato
        return (precision, recall)
    else:  # altrimenti accodo il risultato che ho trovato
        return coda.put((precision, recall))


def confusion_matrix_par(n_test, tipo, N, ID_test, risultati):
    coda = mp.Queue()
    threads = [threading.Thread(target=confusion_matrix, args=(pos_utente, n_test[pos_utente]), #storia
                                kwargs={"tipo": tipo, "N": N, "ID_test": ID_test, "risultati": risultati, "coda": coda})
               for pos_utente in range(len(n_test))]
    for t in threads:
        t.start()
    x = [coda.get() for t in threads]
    for t in threads:
        t.join()  # blocca il MainThread finché t non è completato
    return x
