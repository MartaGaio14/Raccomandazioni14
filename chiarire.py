import urllib
from bs4 import BeautifulSoup
import pandas
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import csv
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

news.loc[news["ID"] == "N113363"]
news=news.drop(46236)

news_file.close()


news2={}
news2=pandas.DataFrame(news2)
for i in tqdm.tqdm(range(0,len(S_norep))): 
    a=news.loc[news["ID"] == S_norep[i]]
    news2=pandas.concat([news2, a], ignore_index=True)

print(news2)
news2.info(null_counts=True)

 
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
    
with open("testi.csv", "w") as file:
    writer=csv.writer(file)
    for i in tqdm.tqdm(range(0, len(news2))):
        url=news2.URL[i]
        html = urllib.request.urlopen(url)
        testo=vattene(html)
        writer.writerow([news2.ID[i], testo])
        


#apertura file testi
csv_file = open("testi.csv", encoding="Latin1")
read = pandas.read_csv(csv_file, sep=",", header=None, names=["ID", "Testo"])
print(read.head())

csv_file.close()
        