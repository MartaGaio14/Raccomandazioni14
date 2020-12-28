
# map
# separo le parole e tolgo i simboli speciali
# ad ogni parola(chiave) associo un argomento=1
# group
# ordino le parole
# ad ogni chiave associo una lista di uno quante sono le volte che compare la parola
# reduce
# sommo il numero di volte che compare la parola
# stampo le coppie parola-contatore


    ####################################

from mrjob.job import MRJob
import requests
from bs4 import BeautifulSoup
import time







import pandas
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

prova = pandas.read_csv("cose/testi_train.csv", names=["ID", "Testo"], header=None, error_bad_lines=False)
prova = prova[:1000]

stop_words = set(stopwords.words("english"))
lettere = list('abcdefghijklmnopqrstuvwxyz')
numeri = list('0123456789')
for i in range(0, len(lettere)):
    stop_words.add(lettere[i])
for i in range(0, len(numeri)):
    stop_words.add(numeri[i])
stop_words.add("getty")
stop_words.add("slides")
verbi_comuni = "ask become begin call come could find get give go hear keep know leave let like live look make may might move need play put run say see seem show start take tell think try use want work would said got made went gone knew known took token saw seen came thought gave given found told left"
verbi_comuni = verbi_comuni.split(" ")
for i in range(len(verbi_comuni)):
    stop_words.add(verbi_comuni[i])

#ad ogni parola toglie se alla fine ci sono caratteri non alfanum o spazi
def clean_word(word):
    return re.sub(r'[^\w]', '', word).lower()

#restituisce True se la parola non è nelle stopwords e se è composta da caratteri alfanumerici (no punteggiatura)
def word_not_in_stopwords(word):
    return word not in stop_words and word and word.isalpha()

REM = ['CC', 'CD', 'DT', 'EX', 'IN', 'LS', 'MD', 'PDT', 'POS', 'PRP', 'PSRP$', 'RB', 'RBR', 'RBS', 'TO', 'UH',
           'WDT', 'WP', 'WPD$', 'WRB', 'RP']
# restituisce True se la parola non è da rimuovere data la sua funzione grammaticale
def part_of_speach_tagging(word):
    tag = nltk.pos_tag([word])
    tag = list(tag[0])[1] #estrae il codice POS della parola
    for i in REM:
        if tag == i:
            return False
    return True

#lemming e stemming
stemmer = PorterStemmer()
def stem(word):
    return stemmer.stem(word)

#The mapper gets a text, splits it into tokens, cleans them and filters stop words and non-words, finally,
#it counts the words within this single text document
import collections
def mapper(text):
    tokens_in_text = text.split()
    tokens_in_text = map(clean_word, tokens_in_text)
    tokens_in_text = filter(word_not_in_stopwords, tokens_in_text)
    tokens_in_text = filter(part_of_speach_tagging, tokens_in_text)
    tokens_in_text = map(stem, tokens_in_text)
    return list(tokens_in_text) # testo preprocessato

#The reducer function gets 2 counters and merges them
def reducer(text1, text2):
    return text1+text2

def chunkify(list, n):
    for i in range(0, len(list), n):
        yield list[i: i + n]

from functools import reduce
data_chunks = chunkify(prova.Testo, 4)

# the chunk_mapper gets a chunk and does a MapReduce on it
def chunk_mapper(chunk):
    mapped = map(mapper, chunk)
    mapped = zip(chunk, mapped)
    reduced = reduce(reducer, mapped)
    return reduced

import time
inizio=time.time()
#step 1:
mapped = map(chunk_mapper, data_chunks)
#step 2:
reduced = reduce(reducer, mapped)
fine=time.time()
print(fine-inizio)



from mrjob.job import MRJob
from preprocessing import *


class prep(MRJob):
    def mapper(self, _, line): #_=chiave, line=valore
        # yield each word in the line
        chiave, valore = line.split("\t")
        if valore == "":
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