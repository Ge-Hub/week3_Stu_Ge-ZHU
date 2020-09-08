
import pandas as pd 
import numpy as np
import jieba
import time

start = time.time()

#Step1，数据加载 加载sqlResult.csv及停用词chinese_stopwords.txt
with open ('chinese_stopwords.txt', 'r', encoding='utf-8') as file:
    stopwords = [i[:-1] for i in file.readlines()]
    #数据加载&缺失值处理
news = pd.read_csv('sqlResult.csv', encoding = 'gb18030')
#print(news.head())
print(news.shape)
print(news[news.content.isna()].head())

print("-"*88)
#Step2，数据预处理
    #1）数据清洗，针对content字段为空的情况，进行dropna
news = news.dropna(subset=['content'])
print(news.shape)
print("-"*88)
    #2）分词，使用jieba进行分词
import jieba
    #3）将处理好的分词保存到 corpus.pkl，方便下次调用
def split_text(text):
    text = text.replace(' ','').replace('\n','')
    
    #str.strip([chars]) chars -- 移除字符串头尾指定的字符序列。
        # 以上下例演示了只要头尾包含有指定字符序列中的字符就删除：
            #str = "123abcrunoob321"
            #print (str.strip( '12' ))  # 字符序列为 12
        # 去除首尾字符 0: print str.strip( '0' )
        # 去除首尾空格 : print str.strip()

    text2 = jieba.cut(text.strip())
    #去停用词
    result = ''.join([w for w in text2 if w not in stopwords])
    return result
    # loc——通过行标签索引行数据 /   loc[1]表示索引的是第1行（index 是整数）; loc[‘d’]表示索引的是第’d’行（index 是字符）/ loc扩展——索引某列 df.loc[:,['c']]  
    # iloc——通过行号获取行数据 /  iloc索引列数据 df.iloc[:,[1]] / 想要获取哪一行就输入该行数字 df.loc[1] 
    # 预浏览加分词
print(news.iloc[0].content)
print(split_text(news.iloc[0].content))
    #创建文库，通过创建list。对所有文本进行分词后，放到list里。
    # map() 会根据提供的函数对指定序列做映射。map函数的用法，需要传2个参数，
    # 在python内部，就相当于一个for循环，把每次遍历的那个数字交给这个处理方法map(一个处理方法，一个可迭代对象)https://www.cnblogs.com/gouguoqilinux/p/9153514.html

corpus = list(map(split_text, [str(i) for i in news.content]))
print(corpus[0])
print(len(corpus))

    #保存到文件：ickle模块实现了基本的数据序列化pickle.dump(obj, file, [,protocol]) / 与反序列化 pickle.load(file)
    #序列化对象，将对象obj保存到文件file中去。参数protocol是序列化模式，默认是0（ASCII协议，表示以文本的形式进行序列化），
    #protocol的值还可以是1和2（1和2表示以二进制的形式进行序列化。其中，1是老式的二进制协议；2是新二进制协议）。file表示保存到的类文件对象，file必须有write()接口，file可以是一个以'w'打开的文件或者是一个StringIO对象，也可以是任何可以实现write()接口的对象。
import pickle
with open('corpus.pkl', 'wb') as file:
    pickle.dump(corpus, file)

print("-"*118)
#Step3，计算corpus中的TF-IDF矩阵。用于评估一个词语对于一个文集或一个语料库中的其中一份文档的重要程度。
    # TF-IDF（term frequency–inverse document frequency）是一种用于信息检索（information retrieval）与文本挖掘（text mining）的常用加权技术
    # 先做TF，再做IDF => TF-IDF TF 和 IDF 以后，将这两个值相乘，就得到了一个词的TF-IDF值。某个词对文章的重要性越高，它的TF-IDF值就越大。
    # https://scikit-learn.org/stable/modules/feature_extraction.html
    #CountVectorizer：只考虑词汇在文本中出现的频率
    #TfidfVectorizer：除了考量某词汇在文本出现的频率，还关注包含这个词汇的所有文本的数量；能够削减高频没有意义的词汇出现带来的影响, 挖掘更有意义的特征
    # https://www.cnblogs.com/Lin-Yi/p/8974108.html 
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
countvectorizer = CountVectorizer(encoding='gb18030', min_df=0.015)
tfidftransformer = TfidfTransformer()

countvector = countvectorizer.fit_transform(corpus)
tfidf = tfidftransformer.fit_transform(countvector)

print("-"*118)
#Step4，预测文章风格是否和自己一致 使用分类模型（比如MultinomialNB），对于文本的特征（比如TF-IDF）和label（是否为新华社）进行训练
    #标记是否自己的新闻
label = list(map(lambda source:1 if '新华' in str(source) else 0, news.source))
    #数据集切分，70%训练集，30%测试集
    # X_train,X_test, y_train, y_test =sklearn.model_selection.train_test_split(train_data,train_target,test_size=0.4, random_state=0,stratify=y_train) https://www.cnblogs.com/Yanjy-OnlyOne/p/11288098.html 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(tfidf.toarray(), label, test_size=0.3)

    # The multinomial Naive Bayes classifier is suitable for classification with discrete features 
    # (e.g., word counts for text classification)
    # https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html 
    # https://blog.csdn.net/YangWei_19/article/details/79971257 

from sklearn.naive_bayes import MultinomialNB
model_1 = MultinomialNB()
model_1.fit(X_train, y_train)
y_predict = model_1.predict(X_test)

from sklearn.metrics import accuracy_score, precision_score, recall_score

print('Accuracy_Score:', accuracy_score(y_test, y_predict))
print('Precision_Score:', precision_score(y_test, y_predict))
print('Recall_Score:', recall_score(y_test, y_predict))

print("-"*118)

from sklearn.naive_bayes import BernoulliNB
model_2 = BernoulliNB()
model_2.fit(X_train, y_train)
y_predict = model_2.predict(X_test)

print('Accuracy_Score:', accuracy_score(y_test, y_predict))
print('Precision_Score:', precision_score(y_test, y_predict))
print('Recall_Score:', recall_score(y_test, y_predict))

print("-"*118)

from sklearn.naive_bayes import GaussianNB
model_3 = GaussianNB()
model_3.fit(X_train, y_train)
y_predict = model_3.predict(X_test)

print('Accuracy_Score:', accuracy_score(y_test, y_predict))
print('Precision_Score:', precision_score(y_test, y_predict))
print('Recall_Score:', recall_score(y_test, y_predict))

print("-"*118)
#Step5，找到可能Copy的文章，即预测label=1，但实际label=0
    #使用模型检测抄袭新闻，预测风格
prediction = model_2.predict(tfidf.toarray())
labels = np.array(label)
    #compare_news_index 有两列，prediction为预测风格, labels为真实新华社
compare_news_index = pd.DataFrame({'prediction': prediction, 'labels': labels})
    # 怀疑对象，prediction = 1, 真实为0
copy_news_index = compare_news_index[(compare_news_index['prediction']==1) & (compare_news_index['labels']==0)]
    #实际为新华社的新闻
xinhuashe_news_index = compare_news_index[(compare_news_index['labels']==1)].index
    #抄袭数量过多，会产生噪音。这里只是拿新华社风格做参考。是否抄袭其实是与文章相关，不是风格相关。
print('可能为copy的新闻条数', len(copy_news_index))

print("-"*118)
#Step6，根据模型预测的结果来对全量文本进行比对，如果数量很大，我们可以先用k-means进行聚类降维，比如k=25种聚类 
    #使用Kmeans对文章进行聚类 。向量之间距离是余玄距离，通过Normalizer fit_transform 把余玄距离转换到欧式距离， 再使用kMeans.
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
normalizer = Normalizer()
scaled_array= normalizer.fit_transform(tfidf.toarray())
    #使用kMeans，全量文档进行聚类
kmeans = KMeans(n_clusters=8)
k_labels = kmeans.fit_predict(scaled_array)
print(k_labels.shape)
 
    #创建id_class 每个ID等于哪个分类
id_class = {index:class_ for index, class_ in enumerate(k_labels)}
    #创建空集合的字典 from collections import defaultdict, defaultdict的作用是在于，当字典里的key不存在但被查找时，返回的不是keyError而是一个默认值
from collections import defaultdict
class_id = defaultdict(set)
for index, class_ in id_class.items():
    #只统计新华社发布的class_id
    if index in xinhuashe_news_index.tolist():
        class_id[class_].add(index)

print("-"*118)
# Step7，找到一篇可能的Copy文章，从相同label中，找到对应新华社的文章，并按照TF-IDF相似度矩阵，从大到小排序，取Top10
from sklearn.metrics.pairwise import cosine_similarity
#查找相似文本
from sklearn.metrics.pairwise import cosine_similarity
def find_similar_text(cpindex, top=20):
    #只在新华社发布的文章中找(class id), 余玄向度，聚类方式去完成聚类
    dist_dict = {i:cosine_similarity(tfidf[cpindex], tfidf[i]) for i in class_id[id_class[cpindex]]}
    #从大到小进行排序
    return sorted(dist_dict.items(), key=lambda x:x[1], reverse=True)[:top]

cpindex = 5188
print('availible in Xinhua news:',cpindex in xinhuashe_news_index)
print('availible in Xinhua copy_news:',cpindex in copy_news_index)

similar_list = find_similar_text(cpindex)
print(similar_list)
print('怀疑抄袭\n', news.iloc[cpindex].content)
    #找一篇相似的原文
similar2 = similar_list[0][0]
print('相似原文：\n', news.iloc[similar2].content)


end = time.time()
print("用时：", end-start)




