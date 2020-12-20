

#map
#separo le parole e tolgo i simboli speciali
#ad ogni parola(chiave) associo un argomento=1
#group
#ordino le parole
#ad ogni chiave associo una lista di uno quante sono le volte che compare la parola
#reduce
#sommo il numero di volte che compare la parola
#stampo le coppie parola-contatore


    ####################################

from mrjob.job import MRJob
import requests
from bs4 import BeautifulSoup
import time

def extraction(url):
    r = requests.get(url, timeout=10)
    if r.status_code == 200:
        soup = BeautifulSoup(r.text, 'html.parser')
        sec=soup.find_all('section')
        if len(sec) > 2:
            body_text = sec[2].text.strip()
        else:
            slides = soup.find_all("div", class_="gallery-caption-text")
            body_text = ""
            for i in range(len(slides)):
                body_text += (slides[i].text.strip())
    return body_text



class webscrap(MRJob):
    def mapper(self, _, url): #_=chiave, line=valore
        # yield each word in the line
        yield (1, extraction(url)) #man mano che ciclo creo la coppia chiave-valore

 #passo intermedio group(combiner) è di default
 #capisce già da solo che l'input del reducer è l'output del mapper ma con coppie già aggregate di chiave-valore (valore=lista di valori associati a quella chiave)
    def reducer(self, key, values):
        yield key, values
        #restituisce coppire chiave, valore


if __name__ == '__main__':
    inizio=time.time()
    webscrap.run()
    fine=time.time()
    print(fine-inizio)

