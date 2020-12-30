from collections import defaultdict
from gensim import corpora


# freq = []
# for testo in testi:
#     freq.append(CountFreq(testo))
# # teniamo solo le parole che si ripetono più di una volta
# corpus = [[parola for parola in testo_diz.keys() if testo_diz[parola] > 1] for testo_diz in freq]


def LDA_corpus(testi):
    frequency = defaultdict(int)
    for text in testi:
        for token in text:
            frequency[token] += 1
    # teniamo solo le parole che si ripetono più di una volta
    processed_corpus = [[token for token in text if frequency[token] > 1] for text in testi]
    # a ogni parola associamo un numero
    dictionary = corpora.Dictionary(processed_corpus)
    # a ogni numero corrispondente alle parole si associa la frequenza
    corpus = [dictionary.doc2bow(text) for text in processed_corpus]
    return corpus, dictionary

