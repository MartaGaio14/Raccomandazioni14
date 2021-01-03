from collections import defaultdict

def profilo(storia, diz_grande):
   testi=[]#lista delle rappresentazioni concatenate di tutte le news lette dall'utente
   for articolo in storia:  # selezioniamo solo le news in storia
       testi.extend(diz_grande[articolo])
   t = list(zip(*testi))  # separiamo le tuple(etichetta, peso)
   etichetta=list(t[0])
   peso=list(t[1])
   #in un dizionario salviamo (etichetta, somme dei pesi associati all'etichetta che compaiono in testi)
   profilo = defaultdict(int)
   for j in range(len(etichetta)):
        profilo[etichetta[j]] += peso[j]
    #dividiamo ciascun valore del dizionario per len(storia) per ottenere la media
   chiavi=list(profilo.keys())
   for chiave in chiavi:
       profilo[chiave] /= len(storia)
    #teniamo solo i primi 1000 elemeti della rappresentazione (utile per tfidf)
   if len(profilo)>1000:
        profilo=dict(sorted(profilo.items(), key=lambda item: item[1])[0:1000])
   return profilo