# funzioni per calcolare la similarità
import numpy as np
from numpy import linalg as LA

def compara(dictU, dictA):
    paroleu=dictU.keys()
    parolea=dictA.keys()
    paroleu_set = set(paroleu)
    intersection = list(paroleu_set.intersection(parolea))
    utente=[]
    articolo =[]
    for i in range(len(intersection)):
        utente.append(dictU[intersection[i]])
        articolo.append(dictA[intersection[i]])
    return utente, articolo

def cosSim(dictU, dictA):
    (a, b) = compara(dictU, dictA)
    num = np.dot(a, b)
    # c = [v**2 for v in dictU.values()]
    # d = [v**2 for v in dictA.values()]
    den = LA.norm(list(dictU.values()))*LA.norm(list(dictA.values()))
    return num/den

