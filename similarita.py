# funzioni per calcolare la similarità
import numpy as np
from numpy import linalg as LA


def compara(dictU, dictA):
    utente = []
    articolo = []
    for keyU in dictU:
        for keyA in dictA:
            if keyU == keyA:
                utente.append(dictU[keyA])
                articolo.append(dictA[keyA])
    return utente, articolo

def cosSim(dictU, dictA):
    (a, b) = compara(dictU, dictA)
    num = np.dot(a, b)
    # c = [v**2 for v in dictU.values()]
    # d = [v**2 for v in dictA.values()]
    den = LA.norm(list(dictU.values()))*LA.norm(list(dictA.values()))
    return num/den


