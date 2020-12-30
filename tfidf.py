#TFIDF
import numpy as np
import tqdm

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


def IDF(texts_train):
    N=len(texts_train)
    allwords=[]#lista parole singole del corpus del training set con ripetizioni
    for z in range(0, len(texts_train)):
        norep=np.unique(texts_train[z])
        allwords.extend(norep)
    n_i=CountFreq(allwords) #numero di documenti che contengono un termine
    chiavi=list(n_i.keys())
    for i in range(len(n_i)):
        n_i[chiavi[i]]=np.log10(N/n_i[chiavi[i]])
    return n_i

# def TFIDF(texts, idf):
#     tot_doc = []  # lista di dizionari freq per ogni documento
#     for z in range(0, len(texts)):
#         b = CountFreq(texts[z])
#         tot_doc.append(b)
#     tfidf_corpus = []
#     for j in tqdm.tqdm(range(0,len(tot_doc))):#j è il documento
#         k=list(tot_doc[j].keys()) #lista delle parole nel documento j
#         tfidf_doc=[]
#         for i in range(len(tot_doc[j])):# i è la parola nel documento j
#             max_f=max(list(tot_doc[j].values())) #parola con massima freq nel documento j
#             tf=tot_doc[j][k[i]]/max_f #numero di occorrenze del termine i nel documento j/max_f
#             if k[i] in list(idf.keys()):
#                 tfidf_doc.append([k[i], tf*idf[k[i]] ])
#             else:
#                 tfidf_doc.append([k[i], 0])
#         tfidf_doc=dict(tfidf_doc)
#         if len(tfidf_doc)>750:
#             tfidf_corpus.append(dict(sorted(tfidf_doc.items(), key=lambda item: item[1], reverse=True)[0:750]))
#             #ordina gli elementi del dizionario e la chiave di ordinamento è il peso tfidf (cioè item[1] nella coppia chiave-valore)
#         else:
#             tfidf_corpus.append(tfidf_doc)
#     return tfidf_corpus


#########questa su un testo solo (per parallelizzare)
def TFIDF(testo, idf):
    doc = CountFreq(testo)
    k=list(doc.keys()) #lista delle parole
    tfidf_doc=[]
    for i in range(len(doc)):# i è una parola nel documento
        max_f=max(list(doc.values())) #parola con massima freq nel documento
        tf=doc[k[i]]/max_f #numero di occorrenze del termine i nel documento /max_f
        if k[i] in list(idf.keys()):
            tfidf_doc.append([k[i], tf*idf[k[i]]])
        else:
            tfidf_doc.append([k[i], 0])
    tfidf_doc = dict(tfidf_doc)
    if len(tfidf_doc)>750:
        tfidf_doc = sorted(tfidf_doc.items(), key=lambda item: item[1], reverse=True)[0:750]
        #ordina gli elementi del dizionario e la chiave di ordinamento è il peso tfidf (cioè item[1] nella coppia chiave-valore)
    return tfidf_doc







# allwords = []
# for z in range(0, len(testi_train)):
#     norep = np.unique(testi_train[z])
#     allwords.extend(norep)
# n_i = CountFreq(allwords)
# N = len(testi_train)
# num = 750
# doc_tfidf = TFIDF_par(testi_train, N, n_i, num)
