#TFIDF
import numpy as np
import tqdm
from collections import defaultdict

##creazione del corpus
# frequenza di ogni parola
def CountFreq(word_list):
    word_dict = {}
    for word in word_list:
        if word not in word_dict:
            word_dict[word] = 1
        else:
            word_dict[word] += 1
    return word_dict

def IDF(testi_train, testi_test):
    frequency = defaultdict(int)
    N = len(testi_train)
    for testo in testi_train:
        for parola in testo:
            frequency[parola] += 1
    idf = defaultdict(int)
    chiavi = frequency.keys()
    for chiave in chiavi:
            idf[chiave] = np.log10(N/frequency[chiave])
    parole_train = idf.keys()
    allwords = []  # lista parole singole del corpus del training set con ripetizioni
    for testo in testi_test:
        allwords.extend(testo)
    parole_test = list(dict.fromkeys(allwords))
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
    for j in tqdm.tqdm(range(0,len(tot_doc))):#j è il documento
        k=list(tot_doc[j].keys()) #lista delle parole nel documento j
        tfidf_doc=[]
        for i in range(len(tot_doc[j])):# i è la parola nel documento j
            max_f=max(list(tot_doc[j].values())) #parola con massima freq nel documento j
            tf=tot_doc[j][k[i]]/max_f #numero di occorrenze del termine i nel documento j/max_f
            tfidf_doc.append([k[i], tf*idf[k[i]] ])
        tfidf_doc=dict(tfidf_doc)
        if len(tfidf_doc)>750:
            tfidf_corpus.append(dict(sorted(tfidf_doc.items(), key=lambda item: item[1], reverse=True)[0:750]))
            #ordina gli elementi del dizionario e la chiave di ordinamento è il peso tfidf (cioè item[1] nella coppia chiave-valore)
        else:
            tfidf_corpus.append(tfidf_doc)
    return tfidf_corpus

#########questa su un testo solo (per parallelizzare)
#
# def TFIDF(testo, idf):
#     doc = CountFreq(testo)
#     k=list(doc.keys()) #lista delle parole
#     tfidf_doc=[]
#     for i in range(len(doc)):# i è una parola nel documento
#         i=0
#         max_f=max(list(doc.values())) #parola con massima freq nel documento
#         tf=doc[k[i]]/max_f #numero di occorrenze del termine i nel documento /max_f
#         if k[i] in list(idf.keys()):
#             tfidf_doc.append([k[i], tf*idf[k[i]]])
#         else:
#             tfidf_doc.append([k[i], 0])
#     tfidf_doc = dict(tfidf_doc)
#     if len(tfidf_doc)>750:
#         tfidf_doc = sorted(tfidf_doc.items(), key=lambda item: item[1], reverse=True)[0:750]
#         #ordina gli elementi del dizionario e la chiave di ordinamento è il peso tfidf (cioè item[1] nella coppia chiave-valore)
#     return tfidf_doc

