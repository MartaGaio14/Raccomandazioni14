#funzioni per calcolare la similarità

def compara(dictU, dictA):
    utente=[]
    articolo=[]
    for keyU in dictU:
        for keyA in dictA:
            if keyU == keyA:
                utente.append(dictU[keyA])
                articolo.append(dictA[keyA])
    return utente,articolo
            
def CosDist(u, v):
    dist = np.dot(u, v) / (LA.norm(u) * LA.norm(v))
    return dist

def similarità(dictU, dictA):
    (a,b)=compara(dictU, dictA)
    return CosDist(a,b)