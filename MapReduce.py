from mrjob.job import MRJob
from preprocessing import preprocessing1

class prep(MRJob):
    def mapper(self, _, line): #_=chiave, line=valore
        # yield each word in the line
        chiave, valore = line.split("\t")
        if valore == "sbagliata":
            yield (chiave, 0)
        else:
            for word in preprocessing1(valore): #prende il testo in ingresso e lo dividiamo parola per parola
                yield (chiave, word) #man mano che ciclo creo la coppia chiave-valore

    #passo intermedio group(combiner) è di default
    #capisce già da solo che l'input del reducer è l'output del mapper ma con coppie già aggregate di chiave-valore (valore=lista di valori associati a quella chiave)
    def reducer(self, key, values):
        yield (key, list(values)) #restituisce coppie chiave, valore


if __name__ == '__main__':
    prep.run()