import tqdm
import multiprocessing as mp
import os
import threading


def ContentBasedProfile(Hist_0, dimensioni, pesi):
    #lista di liste di due elementi ciascuna: numero del topic+somma dei pesi corrispondenti
    somme=[] 
    b=[] #lista delle dimensioni già viste
    for i in range(len(dimensioni)):
        if dimensioni[i] not in b:
            b.append(dimensioni[i]) 
            a=[dimensioni[i], pesi[i]] 
            somme.append(a)
        else:
            for j in range(len(b)):
                if dimensioni[i]==b[j]:
                    somme[j][1]+=pesi[i] #somma al peso corrispondente alla dimensione
    for s in range(len(somme)):
         #divido pesi relativi a ciascuna parola per il n di news lette da ciascun utente
        p=somme[s][1]/len(Hist_0) 
        somme[s][1]= p
    return dict(somme)

####################CONTENT BASED PROFILE IN RAPPRESENTAZIONE TFIDF
    
#doc_tfidf lista di dizionari con le rappresentazioni in tfidf di tutte le news
#Hist lista di liste delle news lette da ciascun utente
#S_norep lista intera delle news di training lette dalla totalità degli utenti

#parlallelizziamo
    


#calcola il profilo di un utente 
def utenti_tfidf(Hist_u, doc_tfidf, S_norep, coda= None):
   testi=[]#lista di dizionari, ogni dizionario è una news in tfidf letta dall'utente
        #in questione (l'iesimo)
   for j in range(len(doc_tfidf)):
       if S_norep[j] in Hist_u:
           testi.append(doc_tfidf[j])
   diz={}
        #dimensioni-->lista di dimensioni (riferita a tutte le news salvate in testi)
        #pesi-->lista dei pesi corrispondenti (i pesi sono le chiavi del dizionario)
   for t in range(len(testi)):
        pesi=list(testi[t].values())
        dimensioni=list(testi[t].keys())
        for j in range(len(testi[t])):
            diz[dimensioni[j]]=pesi[j]
   profilo=dict(ContentBasedProfile(Hist_u,list(diz.keys()),list(diz.values())))
   if len(profilo)>1000:
        profilo=dict(sorted(profilo.items(), key=lambda item: item[1])[0:1000])
   if coda is None:#se è non valorizzato ritorno il profilo
        return profilo
   else:#altrimenti accodo il profilo che ho trovato
        coda.put(profilo)
        
        

def utenti_tfidf_par(Hist, doc_tfidf, S_norep):
    coda = mp.Queue()
    threads = [threading.Thread(target=utenti_tfidf, args=(Hist_u,), 
        kwargs={"doc_tfidf": doc_tfidf,"S_norep": S_norep, "coda": coda}) 
               for Hist_u in tqdm.tqdm(Hist)]
    for t in threads2:
        t.start()
    u_profile_tfidf=[coda.get() for t in threads]
    for t in threads:
        t.join() # blocca il MainThread finché t non è completato
    return u_profile_tfidf

coda2 = mp.Queue()
threads2 = [threading.Thread(target=utenti_tfidf, args=(Hist_u,), 
        kwargs={"doc_tfidf": doc_tfidf,"S_norep": S_norep, "coda": coda2}) 
               for Hist_u in tqdm.tqdm(Hist)]
threads2[999].start()
coda2.get()
###################CONTENT BASED PROFILE IN RAPPRESENTAZIONE LDA
#doc_lda lista di dizionari con le rappresentazioni in lda di tutte le news
#Hist lista di liste delle news lette da ciascun utente
  #S_norep lista intera delle news di training lette dalla totalità degli utenti      
    
def utenti_lda(Hist, doc_lda, S_norep):
    u_profile_lda=[]
    for i in tqdm.tqdm(range(len(Hist))): #i gira negli user 
        testi=[]
        for j in range(len(doc_lda)):
            if S_norep[j] in Hist[i]:
                testi.append(doc_lda[j])
        dimensioni=[] #lista di dimensioni (riferita a tutte le news salvate in testi)
        pesi=[] #lista dei pesi corrispondenti
        for i in range(len(testi)):
            t=list(zip(*testi[i])) #separo le tuple(dimensioni, peso)
            dimensioni.extend(list(t[0]))    
            pesi.extend(list(t[1]))    
        u_profile_lda.append(ContentBasedProfile(Hist[i],dimensioni, pesi))
        #creata una lista di dizionari
    return u_profile_lda

