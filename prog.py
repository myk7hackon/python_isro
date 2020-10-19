import glob

import nltk
from nltk import word_tokenize

func = {}
all_words = {}
alli_words = {}
spam = ''
ham = ''
word_features = []
fp = open('new.txt', 'r')
while 1:
    line = fp.readline()
    if not line:
        break
    # print line
    line = str(line)
    # print line[0]
    if line[0] == '0':
        # print pro
        # k=word_tokenize(line[1:])
        # print k
        # alli_words=nltk.FreqDist(k)
        # s=list(all_words.keys())[0:10]
        # print s
        # word_features=list(all_words.keys())[:1
        ham = ham + line[2:]
    # print ham
    # print word_features
    if line[0] == '1':
        # k=word_tokenize(line[1:])
        # all_words=nltk.FreqDist(k)
        # s=list(all_words.keys())[0:10]
        spam = spam + line[2:]
k = word_tokenize(ham)
all_words = nltk.FreqDist(k)
hm = list(all_words.keys())[0:200]
# print hm
k = word_tokenize(spam)
all_words = nltk.FreqDist(k)
sm = list(all_words.keys())[0:200]
# print sm
print(sm, hm)
total = {}
for a in hm:
    total[a] = False
for a in sm:
    total[a] = True
print(total)
from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(total)
from sklearn.feature_extraction.text import TfidfTransformer

tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
docs_new = ['glass repath', 'diagram']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tf_transformer.transform(X_new_counts)
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB().fit(X_train_tf, total.values())
print(clf.predict(X_new_tfidf))
# print clf.score(X_test_tfs,totally.values())
# classifier=nltk.NaiveBayesClassifier.train(X_train_tf)
# print nltk.classify.accuracy(classifier,X_train_tf)
# print total.values()
'''if 'money' in sm:
	print 'yeah'
count=0
for i in func.keys():
	count=count+1
#print count
total=[]
for i in func.keys():
	g=(i,func[i])
	total.append(g)
#print total
import random
def feature(out):	
	return out
random.shuffle(total)
total=[(feature(a),b) for (a,b) in total]
print total
train_set=total[:4000]
test_set=total[4000:]
classifier=nltk.NaiveBayesClassifier.train(train_set)
print "naive basyes accuracy      ",
print nltk.classify.accuracy(classifier,fp.readlines)*100'''
