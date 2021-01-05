import numpy as np
import tqdm
from collections import defaultdict
from gensim import corpora


# LDA
def LDA_corpus(testi):
    frequency = defaultdict(int)
    # creazione dizionario che contiene la frequenza con cui compare ogni parola nell'intero corpus
    for text in testi:
        for token in text:
            frequency[token] += 1
    # vengono tenute solo le parole che si ripetono più di una volta
    processed_corpus = [[token for token in text if frequency[token] > 1] for text in testi]  # lista di liste di parole
    # a ogni parola del corpus associamo un numero identificativo
    dictionary = corpora.Dictionary(processed_corpus)
    # a ogni numero corrispondente alle parole si associa la frequenza
    corpus = [dictionary.doc2bow(text) for text in processed_corpus]
    return corpus, dictionary


# TFIDF

# frequenza di ogni parola
def CountFreq(word_list):
    word_dict = {}
    for word in word_list:
        if word not in word_dict:
            word_dict[word] = 1
        else:
            word_dict[word] += 1
    return word_dict


# IDF calcolato sul corpus di training
# oltre a restituire i valori corrispondenti alle parole presenti nel dataset di training vengono restituite anche
# le parole presenti nel dataset di test e non nel dataset di training, con IDF pari a 0
def IDF(testi_train, testi_test):
    allwords = []  # lista parole non ripetute per ogni documento
    for testo in testi_train:
        norep = list(dict.fromkeys(testo))
        allwords.extend(norep)
    # n_i è un dizionario con chiave: singola parola presente nel corpus, valore: numero di documenti in cui compare la parola
    n_i = CountFreq(allwords)
    N = len(testi_train) # totale dei documenti considerati
    idf = defaultdict(int)
    chiavi = n_i.keys()
    for chiave in chiavi:
        idf[chiave] = np.log10(N / n_i[chiave])
    parole_train = idf.keys()
    allwords1 = []  # lista parole singole del corpus del test set con ripetizioni
    for testo in testi_test:
        allwords1.extend(testo)
    parole_test = list(dict.fromkeys(allwords1)) # lista parole singole del test set senza ripetizioni
    for parola in parole_test:
        if parola not in parole_train:
            idf[parola] = 0
    return idf


def TFIDF(texts, idf):
    tot_doc = []  # lista di dizionari freq per ogni documento
    for z in range(0, len(texts)):
        b = CountFreq(texts[z])
        tot_doc.append(b)
    tfidf_corpus = []
    for j in tqdm.tqdm(range(0, len(tot_doc))):  # j è il documento
        k = list(tot_doc[j].keys())  # lista delle parole nel documento j
        tfidf_doc = []
        for i in range(len(tot_doc[j])):  # i è la parola nel documento j
            max_f = max(list(tot_doc[j].values()))  # parola con massima freq nel documento j
            tf = tot_doc[j][k[i]] / max_f  # numero di occorrenze del termine i nel documento j/max_f
            tfidf_doc.append([k[i], tf * idf[k[i]]])
        tfidf_doc = dict(tfidf_doc)
        if len(tfidf_doc) > 750:
            tfidf_doc = (sorted(tfidf_doc.items(), key=lambda item: item[1], reverse=True)[0:750])
            tfidf_corpus.append(list(tfidf_doc))
            # ordina gli elementi del dizionario e la chiave di ordinamento è il peso tfidf (cioè item[1] nella coppia chiave-valore)
        else:
            tfidf_corpus.append(list(tfidf_doc.items()))
    return tfidf_corpus


