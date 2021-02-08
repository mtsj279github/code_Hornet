
# coding: utf-8
# get_ipython().system('pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pyLDAvis==2.1.2')
# get_ipython().system('pip install -i https://pypi.tuna.tsinghua.edu.cn/simple gensim')
# get_ipython().system('pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tqdm')


import gensim
import warnings
warnings.filterwarnings("ignore")

import os
# os.listdir(".")
import pandas as pd
df=pd.read_excel("2021MCMProblemC_DataSet.xlsx")

# df["Notes"]=df["Notes"].apply(lambda x:str(x).lower())

class_name=df["Lab Status"].unique()



stopwords=[i.strip() for i in open("stopwords.txt").readlines()]


# # LDA

name=class_name[0]
print ("Current lab: \n",name)
tmpdf=df[df["Lab Status"]==name]
tmpdf["new_Notes"]=tmpdf["Notes"].apply(lambda x:[i for i in str(x).split(" ") if i.lower() not in stopwords])

reviews_2=list(tmpdf["new_Notes"])

import gensim
from gensim import corpora
import pyLDAvis
import pyLDAvis.gensim
dictionary = corpora.Dictionary(reviews_2)
doc_term_matrix = [dictionary.doc2bow(rev) for rev in reviews_2]
# Creating the object for LDA model using gensim library
LDA = gensim.models.ldamodel.LdaModel
# Build LDA model
lda_model = LDA(corpus=doc_term_matrix,
                                   id2word=dictionary,
                                   num_topics=5, 
                                   random_state=100,
                                   chunksize=1000,
                                   passes=50)

lda_model.print_topics()

import matplotlib.pyplot as plt
import seaborn as sns
pyLDAvis.enable_notebook()
data=pyLDAvis.gensim.prepare(lda_model, doc_term_matrix, dictionary)
pyLDAvis.display(data)



name=class_name[1]
print ("Current lab: \n",name)
tmpdf=df[df["Lab Status"]==name]
tmpdf["new_Notes"]=tmpdf["Notes"].apply(lambda x:[i for i in str(x).split(" ") if i.lower() not in stopwords])

reviews_2=list(tmpdf["new_Notes"])

import gensim
from gensim import corpora
import pyLDAvis
import pyLDAvis.gensim
dictionary = corpora.Dictionary(reviews_2)
doc_term_matrix = [dictionary.doc2bow(rev) for rev in reviews_2]
# Creating the object for LDA model using gensim library
LDA = gensim.models.ldamodel.LdaModel
# Build LDA model
lda_model = LDA(corpus=doc_term_matrix,
                                   id2word=dictionary,
                                   num_topics=5, 
                                   random_state=100,
                                   chunksize=1000,
                                   passes=50)

lda_model.print_topics()

import matplotlib.pyplot as plt
import seaborn as sns
pyLDAvis.enable_notebook()
data=pyLDAvis.gensim.prepare(lda_model, doc_term_matrix, dictionary)
pyLDAvis.display(data)


name=class_name[2]
print ("Current lab: \n",name)
tmpdf=df[df["Lab Status"]==name]
tmpdf["new_Notes"]=tmpdf["Notes"].apply(lambda x:[i for i in str(x).split(" ") if i.lower() not in stopwords])

reviews_2=list(tmpdf["new_Notes"])

import gensim
from gensim import corpora
import pyLDAvis
import pyLDAvis.gensim
dictionary = corpora.Dictionary(reviews_2)
doc_term_matrix = [dictionary.doc2bow(rev) for rev in reviews_2]
# Creating the object for LDA model using gensim library
LDA = gensim.models.ldamodel.LdaModel
# Build LDA model
lda_model = LDA(corpus=doc_term_matrix,
                                   id2word=dictionary,
                                   num_topics=5, 
                                   random_state=100,
                                   chunksize=1000,
                                   passes=50)

lda_model.print_topics()

import matplotlib.pyplot as plt
import seaborn as sns
pyLDAvis.enable_notebook()
data=pyLDAvis.gensim.prepare(lda_model, doc_term_matrix, dictionary)
pyLDAvis.display(data)

name=class_name[3]
print ("Current lab: \n",name)
tmpdf=df[df["Lab Status"]==name]
tmpdf["new_Notes"]=tmpdf["Notes"].apply(lambda x:[i for i in str(x).split(" ") if i.lower() not in stopwords])

reviews_2=list(tmpdf["new_Notes"])

import gensim
from gensim import corpora
import pyLDAvis
import pyLDAvis.gensim
dictionary = corpora.Dictionary(reviews_2)
doc_term_matrix = [dictionary.doc2bow(rev) for rev in reviews_2]
# Creating the object for LDA model using gensim library
LDA = gensim.models.ldamodel.LdaModel
# Build LDA model
lda_model = LDA(corpus=doc_term_matrix,
                                   id2word=dictionary,
                                   num_topics=5, 
                                   random_state=100,
                                   chunksize=1000,
                                   passes=50)

lda_model.print_topics()

import matplotlib.pyplot as plt
import seaborn as sns
pyLDAvis.enable_notebook()
data=pyLDAvis.gensim.prepare(lda_model, doc_term_matrix, dictionary)
pyLDAvis.display(data)

# # LSI


for i in range(4):
    name=class_name[i]
    print ("Current lab: \n",name)
    tmpdf=df[df["Lab Status"]==name]
    tmpdf["new_Notes"]=tmpdf["Notes"].apply(lambda x:[i for i in str(x).split(" ") if i.lower() not in stopwords])

    reviews_2=list(tmpdf["new_Notes"])
    dictionary = corpora.Dictionary(reviews_2)
    doc_term_matrix = [dictionary.doc2bow(rev) for rev in reviews_2]
    LSI=gensim.models.LsiModel
    lsi_model =LSI(corpus=doc_term_matrix,
                                       id2word=dictionary,
                                       num_topics=5, 
    )
    print (lsi_model.print_topics())


# # textrank  and tfidf 

df["new_Notes"]=df["Notes"].apply(lambda x:[i for i in str(x).split(" ") if i.lower() not in stopwords])


reviews_2=list(df["new_Notes"])


import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
tv = TfidfVectorizer(use_idf=True, smooth_idf=True, norm=None)
train = [" ".join(i) for i in reviews_2]
tv_fit = tv.fit_transform(train)



def text_tfidf(text):
    d=dict(zip(tv.get_feature_names(),list(np.array(tv.transform([text]).todense())[0,:])))
#     print(sorted(d.items(), key = lambda kv:(kv[1], kv[0]),reverse=True)[:10]) 
    key=sorted(d.items(), key = lambda kv:(kv[1], kv[0]),reverse=True)[:10]
    return key


tmptext=" ".join(reviews_2[0])
key_tfidf=text_tfidf(tmptext)



class TextRank(object):

    def __init__(self, sentence, window, alpha, iternum):
        self.word_list = sentence
        self.window = window
        self.alpha = alpha
        self.edge_dict = {}
        self.iternum = iternum

    def cutSentence(self):
        #     jieba.load_userdict('user_dict.txt')
        #     tag_filter = ['a','d','n','v']
        #     seg_result = pseg.cut(self.sentence)
        self.word_list = self.word_list

    #     print(self.word_list)


    def createNodes(self):
        tmp_list = []
        word_list_len = len(self.word_list)
        for index, word in enumerate(self.word_list):
            if word not in self.edge_dict.keys():
                tmp_list.append(word)
                tmp_set = set()
                left = index - self.window + 1
                right = index + self.window
                if left < 0: left = 0
                if right >= word_list_len: right = word_list_len
                for i in range(left, right):
                    if i == index:
                        continue
                    tmp_set.add(self.word_list[i])
                self.edge_dict[word] = tmp_set

    def createMatrix(self):
        self.matrix = np.zeros([len(set(self.word_list)), len(set(self.word_list))])
        self.word_index = {}
        self.index_dict = {}

        for i, v in enumerate(set(self.word_list)):
            self.word_index[v] = i
            self.index_dict[i] = v
        for key in self.edge_dict.keys():
            for w in self.edge_dict[key]:
                self.matrix[self.word_index[key]][self.word_index[w]] = 1
                self.matrix[self.word_index[w]][self.word_index[key]] = 1

        for j in range(self.matrix.shape[1]):
            sum = 0
            for i in range(self.matrix.shape[0]):
                sum += self.matrix[i][j]
            for i in range(self.matrix.shape[0]):
                self.matrix[i][j] /= sum

    # textrank
    def calPR(self):
        self.PR = np.ones([len(set(self.word_list)), 1])
        for i in range(self.iternum):
            self.PR = (1 - self.alpha) + self.alpha * np.dot(self.matrix, self.PR)

            # 输出词和相应的权重

    def printResult(self):
        word_pr = {}
        for i in range(len(self.PR)):
            word_pr[self.index_dict[i]] = self.PR[i][0]
        res = sorted(word_pr.items(), key=lambda x: x[1], reverse=True)
        return res



def get_textrank_key(text):
    tr = TextRank(text, 2, 0.85, 700)
    tr.cutSentence()
    tr.createNodes()
    tr.createMatrix()
    tr.calPR()
    res = tr.printResult()
    return res


from tqdm import tqdm
tfidf_all=[]
textrank_all=[]
for i in tqdm(range(len(reviews_2))):
    tmptext=" ".join(reviews_2[i])
    key_tfidf=text_tfidf(tmptext)
    tfidf_all.append(key_tfidf)
    textrank_all.append(get_textrank_key([i for i in reviews_2[i] if i!=""]))



df["textrank"]=textrank_all



df["tfidf"]=tfidf_all



del df["new_Notes"]



df.to_csv("res.csv",header=True,index=False,encoding="utf-8-sig")

