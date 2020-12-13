#funzioni per calcolare la similarità
import numpy as np
from numpy import linalg as LA

def compara(dictU, dictA):
    utente=[]
    articolo=[]
    for keyU in dictU:
        for keyA in dictA:
            if keyU == keyA:
                utente.append(dictU[keyA])
                articolo.append(dictA[keyA])
    return utente,articolo
            
def CosSim(u, v):
    dist = np.dot(u, v) / (LA.norm(u) * LA.norm(v))
    return dist

def similarità(dictU, dictA):
    (a,b)=compara(dictU, dictA)
    return CosSim(a,b)