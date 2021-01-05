from mrjob.job import MRJob
from preprocessing import preprocessing1


class prep(MRJob):
    def mapper(self, _, line):
        chiave, valore = line.split("\t")  # chiave: ID articolo, valore: testo articolo
        if valore == "sbagliata":
            yield chiave, 0
        else:
            for word in preprocessing1(valore):  # prende il testo in ingresso e lo dividiamo parola per parola
                yield chiave, word  # man mano che ciclo creo la coppia chiave (ID articolo) - valore (parola
                # preprocessata)

    # passo intermedio group(combiner) è di default

    def reducer(self, key, values):
        yield key, list(values)  # restituisce coppie chiave (ID articolo) - valore (lista parole preprocessate
        # dell'articolo)


if __name__ == '__main__':
    prep.run()
