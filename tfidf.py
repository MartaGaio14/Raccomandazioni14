#TFIDF
import numpy as np
import tqdm
import multiprocessing as mp

def CountFreq(word_list):
    word_dict={} 
    for word in word_list:
        if word not in word_dict:
            word_dict[word]=1
        else:
            word_dict[word]+=1
    return word_dict  

def TF_IDF(texts):
    tot_doc=[] #lista di dizionari freq per ogni documento
    allwords=[]#lista parole singole per ogni documento 
    for z in range(0, len(texts)): 
        b=CountFreq(texts[z])
        tot_doc.append(b) 
        norep=np.unique(texts[z])
        allwords.extend(norep)    
    n_i=CountFreq(allwords) #numero di documenti che contengono un termine 
    N=len(tot_doc) #numero di documenti nel corpus
    tfidf_corpus = []
    for j in tqdm.tqdm(range(0,N)):
        k=list(tot_doc[j].keys())
        tfidf_doc=[]
        for i in range(0, len(tot_doc[j])):
            max_f=max(list(tot_doc[j].values())) #parola con massima freq nel documento j
            tf=tot_doc[j][k[i]]/max_f #numero di occorrenze del termine i nel documento j/max_f
            idf=np.log10(N/n_i[k[i]]) #n_i[k[i]] n documenti che contengono termine i
            tfidf_doc.append([k[i], tf*idf])
        tfidf_doc=dict(tfidf_doc)
        if len(tfidf_doc)>1000:
            tfidf_corpus.append(dict(sorted(tfidf_doc.items(), key=lambda item: item[1], reverse=True)[0:1000]))
            #ordina gli elementi del dizionario e la chiave di ordinamento è il peso tfidf (cioè item[1] nella coppia chiave-valore)
        else:
            tfidf_corpus.append(tfidf_doc)
    return tfidf_corpus  

#testo=texts[0]

def TF_IDF2(testo, N, n_i, num):
     #numero di documenti nel corpus
    tot_doc=CountFreq(testo) # dizionario freq di un documento
    #for j in tqdm.tqdm(range(0,N)):
    k=list(tot_doc.keys())
    tfidf_doc={}
    for i in range(0, len(tot_doc)):
        max_f=max(list(tot_doc.values())) #parola con massima freq nel documento j
        tf=tot_doc[k[i]]/max_f #numero di occorrenze del termine i nel documento j/max_f
        idf=np.log10(N/n_i[k[i]]) #n_i[k[i]] n documenti che contengono termine i
        tfidf_doc[k[i]]= tf*idf
    if len(tfidf_doc)>num:
        tfidf_doc=dict(sorted(tfidf_doc.items(), key=lambda item: item[1], reverse=True)[0:num])
        #ordina gli elementi del dizionario e la chiave di ordinamento è il peso tfidf (cioè item[1] nella coppia chiave-valore)
    return tfidf_doc

#TF_IDF2(texts[0],N,n_i, num)

from functools import partial
from itertools import repeat

def TFIDF_par(testi, N, n_i, num):
    N_CPU = mp.cpu_count()
    pool=mp.Pool(processes=N_CPU)
    #f=partial(TF_IDF2, N,n_i,num)
    #t=pool.map(f, testi)
    t=pool.starmap(TF_IDF2, zip(testi, repeat(N), repeat(n_i), repeat(num)))
    pool.close()
    pool.join()
    return t



