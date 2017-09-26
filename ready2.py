from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
import os
tokenizer = RegexpTokenizer(r'\w+')
ps=PorterStemmer()
import string
import glob
import bleach
import re
from bs4 import BeautifulSoup
for filename in glob.glob('*.txt'):
	f=open(filename,"r")
	p=''
	owl=str(f.readlines())
	owl = BeautifulSoup(owl,"html5lib").text
	k=tokenizer.tokenize(owl)
	htao=['http','html','tcharset','content','www','com','type','text','plain','subject']
	k=[ps.stem(w) for w in k]
	stop_words=set(stopwords.words('english'))
	k=[w for w in k if w.isalpha()]
	k=[w for w in k if len(w)>2]
	for i in k:
		if i not in stop_words and i not in htao:
			p=p+i+' '
	#p=p.translate(None,string.punctutation)
	p=p.lower()
	print 1,
	print '   ',
	print p
	print '\n'
"""import re
import string
import operator
frequency{}
match_pattern = re.findall(r'\b[a-z]{3,15}\b', p)
 
for word in match_pattern:
    count = frequency.get(word,0)
    frequency[word] = count + 1
#print sorted(frequency, key=frequency.__getitem__,reverse='True')
#print sorted_x
for i in frequency.keys():
	if frequency[i]>2:
		print i,
#print frequency.keys()
#print frequency"""

		

