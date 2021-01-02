from collections import defaultdict
# storia=n_train[0]
# lista_tuple=doc_lda_train
# ID=ID_train
def profilo(storia, lista_tuple):
   testi=[]#lista delle rappresentazioni concatenate di tutte le news lette dall'utente
   for articolo in storia: # selezioniamo solo le news in storia
       testi.extend(lista_tuple[articolo])
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
# calcola il profilo content based di un utente u
# storia : History dell'utente u
# lista_tuple : lista con le rappresentazioni (tfidf o lda) delle news in storia. Ogni rappresentazione contiene una
# lista di tuple (etichetta, peso). Per tfidf le etichette sono le parole, per lda il numero del topic
# ID : lista degli ID delle news le cui rappresentazioni sono in lista_tuple

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
       profilo[chiave] /= len(Hist_u)
    #teniamo solo i primi 1000 elemeti della rappresentazione (utile per tfidf)
   if len(profilo)>1000:
        profilo=dict(sorted(profilo.items(), key=lambda item: item[1])[0:1000])
   return profilo



import tqdm
import multiprocessing as mp
import threading
# def ContentBasedProfile(Hist_0, dimensioni, pesi):
#     somme=[]  #lista di liste di due elementi ciascuna: numero del topic e somma dei pesi corrispondenti
#     b=[] #lista delle dimensioni già viste
#     for i in range(len(dimensioni)):
#         if dimensioni[i] not in b:
#             b.append(dimensioni[i])
#             a=[dimensioni[i], pesi[i]]
#             somme.append(a)
#         else:
#             for j in range(len(b)):
#                 if dimensioni[i]==b[j]:
#                     somme[j][1]+=pesi[i] #somma al peso corrispondente alla dimensione
#     # for s in range(len(somme)):
#     #      #divido pesi relativi a ciascuna parola per il n di news lette da ciascun utente
#     #     p=somme[s][1]/len(Hist_0)
#     #     somme[s][1]= p
#     return dict(somme)

####################CONTENT BASED PROFILE IN RAPPRESENTAZIONE TFIDF
    
#doc_tfidf lista di dizionari con le rappresentazioni in tfidf di tutte le news
#Hist lista di liste delle news lette da ciascun utente
#S_norep lista intera delle news di training lette dalla totalità degli utenti


#calcola il profilo di un utente 




# def utenti_tfidf_par(Hist, doc_tfidf, S_norep):
#     coda = mp.Queue()
#     threads = [threading.Thread(target=utenti, args=(Hist_u,),
#         kwargs={"doc_tfidf": doc_tfidf,"ID": S_norep, "coda": coda})
#                for Hist_u in tqdm.tqdm(Hist)]
#     for t in threads:
#         t.start()
#     u_profile_tfidf=[coda.get() for t in threads]
#     for t in threads:
#         t.join() # blocca il MainThread finché t non è completato
#     return u_profile_tfidf

###################CONTENT BASED PROFILE IN RAPPRESENTAZIONE LDA
#doc_lda lista di dizionari con le rappresentazioni in lda di tutte le news
#Hist lista di liste delle news lette da ciascun utente
  #S_norep lista intera delle news di training lette dalla totalità degli utenti      
    
# def utenti_lda(Hist, doc_lda, S_norep):
#     u_profile_lda=[]
#     for i in tqdm.tqdm(range(len(Hist))): #i gira negli user
#         testi=[]
#         for j in range(len(doc_lda)):
#             if S_norep[j] in Hist[i]:
#                 testi.append(doc_lda[j])
#         dimensioni=[] #lista di dimensioni (riferita a tutte le news salvate in testi)
#         pesi=[] #lista dei pesi corrispondenti
#         for i in range(len(testi)):
#             t=list(zip(*testi[i])) #separo le tuple(dimensioni, peso)
#             dimensioni.extend(list(t[0]))
#             pesi.extend(list(t[1]))
#         u_profile_lda.append(ContentBasedProfile(Hist[i],dimensioni, pesi))
#         #creata una lista di dizionari
#     return u_profile_lda
