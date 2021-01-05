# funzioni per calcolare la similarità
import numpy as np
from numpy import linalg as LA

#confronta il profilo dell'utente con il profilo dell'item, entrambi espressi come dizionari,
# e restituisce solo i valori (i pesi) delle chiavi comuni
def compara(dictU, dictA):
    paroleu=dictU.keys()
    parolea=dictA.keys()
    paroleu_set = set(paroleu)
    intersection = list(paroleu_set.intersection(parolea)) #termini comuni
    utente=[]
    articolo =[]
    for i in range(len(intersection)):
        utente.append(dictU[intersection[i]])
        articolo.append(dictA[intersection[i]])
    return utente, articolo

#calcolo della cosine similarity
def cosSim(dictU, dictA):
    (a, b) = compara(dictU, dictA)
    num = np.dot(a, b)
    den = LA.norm(list(dictU.values()))*LA.norm(list(dictA.values()))
    return num/den
