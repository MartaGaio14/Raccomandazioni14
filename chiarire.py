# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 18:57:59 2020

@author: gaiom
"""

import urllib
from bs4 import BeautifulSoup
import pandas
import math
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import tqdm



righe=1000
#dataset con i comportamenti degli users
tsv_file = open("behaviors.tsv")
read_tsv = pandas.read_csv(tsv_file, sep="\t", header=None, names=["IID", "UID", "Time", "History", "Imp"], usecols=[1,3])
tsv_file.close()

B=pandas.DataFrame(read_tsv)
read_tsv.info(null_counts=True)
C=B.dropna()
C=C.reset_index(drop=True)
C.info(null_counts=True)

comp=C[0:righe]

Hist=[]
Storie=[]
for t in range(len(comp)):
    Hist.append(word_tokenize(comp.History[t]))
    Storie=Storie+word_tokenize(comp.History[t])
print(Storie)

S_norep = list(dict.fromkeys(Storie))
print()





#dataset con i dati riguardanti gli items
news_file=open("news.tsv", encoding="Latin1")
read_news=pandas.read_csv(news_file, sep="\t", header=None, names=["ID", "Categoria", "SubCategoria", "Titolo", "Abstract", "URL", "TE", "AE"], usecols=[0, 1, 2, 3, 5])
read_news.info(null_counts=True)
news=pandas.DataFrame(read_news)
print(news)

news_file.close()

dic={
     1:news.loc[news["ID"] == S_norep[0]]}



for i in tqdm.tqdm(range(0,len(S_norep))): 
    a=news.loc[news["ID"] == S_norep[i]]
    news2=pandas.concat([news2, a], ignore_index=True)

news2=news2.drop(0)
news2=news2.reset_index(drop=True)
news2.head(10)
 
#estrazione del testo
    
def vattene(html):
    soup = BeautifulSoup(html) # create a new bs4 object from the html data loaded
    # remove all javascript and stylesheet code
    for script in soup(["script", "style"]): 
        script.extract()
    text = soup.get_text()
    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)
    return text
    
t={ }
for i in tqdm.tqdm(range(0, len(news2))):
    url=news2.URL[i]
    html = urllib.request.urlopen(url)
    testo=vattene(html)
    t[news2.ID[i]]=testo