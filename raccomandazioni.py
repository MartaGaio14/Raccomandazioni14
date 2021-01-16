import multiprocessing as mp
import threading

#funzioni utili per il caso 1

# pos_utente è la posizione dell'utente in ID_utente (da 1 a 1000)
# tipo è il tipo di rappresentazione, "lda" o "tfidf"
# N è il numero di news da raccomandare
# ID_test è lista di news candidate alla raccomandazione
def raccomandati(pos_utente, tipo, N, ID_test, risultati):
    # selezioniamo i risultati riguardanti l'utente in posizione pos_utente
    # nota: i risultati sono salvati in ordine per utente
    inizio = pos_utente * len(ID_test)
    fine = inizio + len(ID_test)
    ut = risultati[inizio:fine]
    ut = ut.sort_values(by=[tipo], ascending=False) # ordine decrescente dei dati rispetto alla similarità
    ut = ut.reset_index(drop=True)
    top_tipo = ut[0:N] # le prime N news più simili
    return list(top_tipo.NID)

# storia: lista delle news lette dall'utente (n_test[i])
def confusion_matrix(pos_utente, storia, tipo, N, ID_test, risultati, coda=None):
    racc = raccomandati(pos_utente, tipo, N, ID_test, risultati) # lista delle news raccomandate per l'utente
    racc_set = set(racc)
    intersection = list(racc_set.intersection(storia))  # news lette e raccomandate
    tp = len(intersection)
    fp = N - tp  # false positive
    fn = len(storia) - tp  # false negative
    precision = tp / (tp + fp)  # equivalente a tp/ N
    recall = tp / (tp + fn) # equivalente a tp / len(storia)
    if coda is None:  # se è non valorizzato ritorna il risultato
        return (precision, recall)
    else:  # altrimenti accoda il risultato che ho trovato
        return coda.put((precision, recall))

# funzione parallelizzata della funzione confusion_matrix tramite multithreding
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

#funzioni utili per il caso 2


# pos_utente è la posizione dell'utente in ID_utente (da 1 a 1000)
# tipo è il tipo di rappresentazione, "lda" o "tfidf"
# N è il numero di news da raccomandare
# Impr è lista di news candidate alla raccomandazione per ogni utente, con i rating 0 o 1
def raccomandati2(pos_utente, tipo, N, Impr, risultati):
    # selezioniamo i risultati riguardanti l'utente in posizione pos_utente
    # nota: i risultati sono salvati in ordine per utente
    inizio = 0
    if pos_utente != 0:
        for i in range(pos_utente):
            inizio += len(Impr[i])
    fine = inizio + len(Impr[pos_utente])
    ut = risultati[inizio:fine]  # selezione dati relativi all'utente in questione
    ut = ut.sort_values(by=[tipo], ascending=False)  # ordine decrescente dei dati rispetto alla similarità
    ut = ut.reset_index(drop=True)
    top_tipo = ut[0:N]
    return list(top_tipo.NID)

# funzione che data l'impression di un utente restituisce la lista delle news da lui cliccate
def piaciute(imp):
    true=[]
    for notizia in list(imp.keys()):
        if imp[notizia]== '1':
            true.append(notizia)
    return true

# funzione che restituisce precisione e recall delle raccomandazioni effettuate per un certo utente
def confusion_matrix2(pos_utente, tipo, N, Impr, risultati, coda=None):
    piac = piaciute(Impr[pos_utente]) # lista delle news cliccate dall'utente
    racc = raccomandati2(pos_utente, tipo, N, Impr, risultati) # lista delle news raccomandate per l'utente
    racc_set = set(racc)
    intersection = list(racc_set.intersection(piac))  # news lette e raccomandate
    tp = len(intersection)
    fp = N - tp  # false positive
    fn = len(piac) - tp  # false negative
    precision = tp / (tp + fp)  # equivalente a tp/ N
    recall = tp / (tp + fn) # equivalente a tp / len(storia)
    if coda is None:  # se è non valorizzato ritorna il risultato
        return (precision, recall)
    else:  # altrimenti accoda il risultato che ho trovato
        return coda.put((precision, recall))

def confusion_matrix_par2(n_test, tipo, N, Impr, risultati):
    coda = mp.Queue()
    #args: argomenti dinamici, la posizione dell'utente e la lista delle news del test set che ha letto (storia)
    #kwargs: argomenti statici
    threads = [threading.Thread(target=confusion_matrix2, args=(pos_utente,),
                                kwargs={"tipo": tipo, "N": N, "Impr": Impr, "risultati": risultati, "coda": coda})
               for pos_utente in range(len(n_test))]
    for t in threads:
        t.start()
    x = [coda.get() for t in threads]
    for t in threads:
        t.join()  # blocca il MainThread finché t non è completato
    return x

