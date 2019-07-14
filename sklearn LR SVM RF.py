from lxml import etree
from gensim import corpora,models
from scipy.sparse import csr_matrix
import jieba
import re
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import  metrics
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score
import numpy as np
from pprint import pprint
import time

class achieve:
    # 读xml文档
    def readxml(self,address,textlist):
        xml=etree.parse(address)
        root=xml.getroot()
        taglist=root.xpath('//@label')
        for node in root.xpath('//Sentence[@label]'):
            textlist.append(node.text)
        return taglist

    # ——————————————————————————————————————————————————————————————————————————————————————————————————文本预处理

    # 文本预处理
    def convert_doc_to_wordlist(self,str_doc,cut_all):
        sent_list = str_doc.split('\n')
        sent_list = map(self.rm_char, sent_list) # 去掉一些字符，例如\u3000
        word_2dlist = [self.rm_tokens(jieba.cut(part,cut_all=cut_all)) for part in sent_list] # 分词.去停用词
        word_list = sum(word_2dlist,[])
        return word_list

    # 去掉一些字符，例如\u3000
    def rm_char(self,text):
        text = re.sub('\u3000','',text)
        return text

    # 返回停用词表
    def get_stop_words(self,path='./四川大学机器智能实验室停用词库.txt'):
        # stop_words中，每行放一个停用词，以\n分隔
        file = open(path,'rb').read().decode('utf8').split('\n')
        return set(file)

    # 去掉一些停用次和数字
    def rm_tokens(self,words):
        words_list = list(words)
        stop_words = self.get_stop_words()
        for i in range(words_list.__len__())[::-1]:
            if words_list[i] in stop_words: # 去除停用词
                words_list.pop(i)
            elif words_list[i].isdigit():#如果全都是数字
                words_list.pop(i)
        return words_list

    # ——————————————————————————————————————————————————————————————————————————————————————————————————得到词典

    # 得到字典
    def creatdictionary(self,traintext):
        dictionary = corpora.Dictionary()
        for file in traintext:
            file = self.convert_doc_to_wordlist(file,False)
            dictionary.add_documents([file])
        small_freq_ids = [tokenid for tokenid, docfreq in dictionary.dfs.items() if docfreq < 2 ]
        dictionary.filter_tokens(small_freq_ids)
        dictionary.compactify()
        return dictionary

    # ———————————————————————————————————————————————————————————————————————————————————————————获取tf-idf密集向量

    # 获取tf-idf密集向量
    def gettfidf_matrix(self,dictionary,textlist):
        # count = 0
        bow  = []
        for file in textlist:
            # count += 1
            # if count%100 == 0 :
            #     print('{c} at {t}'.format(c=count, t=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())))
            word_list = self.convert_doc_to_wordlist(file, cut_all=False)
            word_bow = dictionary.doc2bow(word_list)
            bow.append(word_bow)
        tfidf_model = models.TfidfModel(corpus=bow,dictionary=dictionary)
        corpus_tfidf = [tfidf_model[doc] for doc in bow]

        data = []
        rows = []
        cols = []
        line_count = 0
        for line in corpus_tfidf:  # lsi_corpus_total 是之前由gensim生成的lsi向量
            for elem in line:
                rows.append(line_count)
                cols.append(elem[0])
                data.append(elem[1])
            line_count += 1
        tfidf_sparse_matrix = csr_matrix((data,(rows,cols)),shape=(len(textlist),len(dictionary))) # 稀疏向量
        tfidf_matrix = tfidf_sparse_matrix.toarray()  # 密集向量
        return tfidf_matrix

    # ——————————————————————————————————————————————sklearn 用逻辑回归，SVM，随机森林模型实现文本中隐式情感的识别（文本分类）

    # sklearn 逻辑回归 实现文本中隐式情感的识别
    def sklearnLR(self,train_set,train_tag,dev_set,dev_tag):
        logreg = LogisticRegression(C=1,max_iter=100,solver='lbfgs',multi_class='multinomial')
        logreg.fit(train_set,train_tag)
        predict=logreg.predict(dev_set)  #用训练的模型Log来预测测试数据
        print(predict)
        f1 = f1_score(dev_tag, predict,average='micro')
        print("f1:",f1)

    # sklearn svm 实现文本中隐式情感的识别
    def sklearnSVM(self,train_set,train_tag,dev_set,dev_tag):
        clf = svm.LinearSVC(C=0.0665,dual=False) # 使用线性核dual=True,max_iter=1000,loss=‘hinge’ or ‘squared_hinge’,tol=
        clf.fit(train_set,train_tag)
        predict=clf.predict(dev_set)  #用训练的模型Log来预测测试数据
        print(predict)
        f1 = f1_score(dev_tag, predict,average='micro')
        print("f1:",f1)

    # sklearn 随机森林 实现文本中隐式情感的识别
    def sklearnRF(self,train_set,train_tag,dev_set,dev_tag):
        clf = RandomForestClassifier(n_estimators=100,max_features='log2',random_state=1)
        clf.fit(train_set,train_tag)
        predict=clf.predict(dev_set)  #用训练的模型Log来预测测试数据
        print(predict)
        f1 = f1_score(dev_tag, predict,average='micro')
        print("f1:",f1)

def main():
    X=achieve()

    train_text=[]

    train_tag=X.readxml("SMP2019_ECISA_Train.xml",train_text)

    dev_text=[]

    dev_tag=X.readxml("SMP2019_ECISA_Dev.xml",dev_text)

    dictionary=X.creatdictionary(train_text)

    train_set=X.gettfidf_matrix(dictionary,train_text)

    dev_set=X.gettfidf_matrix(dictionary,dev_text)

    X.sklearnLR(train_set,train_tag,dev_set,dev_tag)

    X.sklearnSVM(train_set,train_tag,dev_set,dev_tag)

    X.sklearnRF(train_set,train_tag,dev_set,dev_tag)

# ————————————————————————————————————————————————————————————————————————————————————————————————————————————调用main函数

main()