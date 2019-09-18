from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import nltk
import re
import math
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import random

'''
"""## Connect to Google Drive"""
from google.colab import drive
drive.mount('/content/drive')
'''

"""## Text Pre-Processing"""
## Keep English, digital and space
def get_en_text(text):
    comp = re.compile('( \'\w)|(\w\' )|_|[^\w\' ]+|http')
    text = comp.sub(' ', text).lower()
    return text

## Word tokenization and lemmatization
def get_lemma_tokens(text):
    lem = WordNetLemmatizer()
    tokens = get_en_text(text).split()
    for idx,word in enumerate(tokens):
        tokens[idx] = lem.lemmatize(word, 'v')
        if tokens[idx] != word:
            continue
        tokens[idx] = lem.lemmatize(word, 'n')
        if tokens[idx] != word:
            continue
        tokens[idx] = lem.lemmatize(word, 'a')
        if tokens[idx] != word:
            continue
        tokens[idx] = lem.lemmatize(word, 'r')
        if tokens[idx] != word:
            continue
    return tokens

## Get clean text, remove stopwords
def get_clean_text(text):
    tokens = get_lemma_tokens(text)
    stopWords = set(stopwords.words('english'))
    clean_text = ''
    for token in tokens:
        if token not in stopWords:
            clean_text += ' '+ token
    return clean_text


"""## Process Training Set and Test Set"""
# data_folder = Path("drive/My Drive/SML/")
data_folder = Path("../Data/Original/")
train_file = data_folder / "train_tweets.txt"
labels = []
tweets = []
with open(train_file) as train_tweets:
    lines = train_tweets.read().splitlines()
    for i,l in enumerate(lines, start = 1):
        line_data = l.split('\t')
        tweet = get_clean_text(line_data[1])
        user = line_data[0]
        labels.append(user)
        tweets.append(tweet)
columnsname = ['user','tweet']
alldata = pd.DataFrame(columns=columnsname, data=data)
#alldata.to_csv('drive/My Drive/SML/all_clean_data.csv', encoding='utf-8')
alldata.to_csv('../Data/Processed/all_clean_data.csv', encoding='utf-8')

test_file = data_folder / "test_tweets_unlabeled.txt"
test_text = []
with open(test_file) as test_tweets:
    lines = test_tweets.read().splitlines()
    for l in lines:
        test_text.append(get_clean_text(l))      
test_data = pd.DataFrame(test_text,columns =['tweet'])
#test_data.to_csv('drive/My Drive/SML/test_clean_data.csv', encoding='utf-8')
test_data.to_csv('../Data/Processed/test_clean_data.csv', encoding='utf-8')


"""## Load all training data and divided it into a new training set and test set"""
data = list(zip(labels,tweets)) #328759, befor the deletion of '' it's 328932
data = pd.read_csv('all_clean_data.csv', encoding='utf-8')
all_labels = data['user'].tolist()

random.shuffle(data)
train_test_ratio = 1  #6/10
use_size = math.floor(len(data) * train_test_ratio)
train_size = int(use_size*9/10)
print(use_size)
print(train_size)

traindata = pd.DataFrame(columns=columnsname, data=data[:train_size])
#traindata.to_csv('drive/My Drive/SML/train.csv', encoding='utf-8')
traindata.to_csv('../Data/Processed/train.csv', encoding='utf-8')
testdata = pd.DataFrame(columns=columnsname, data=data[train_size:use_size])
#testdata.to_csv('drive/My Drive/SML/test.csv', encoding='utf-8')
traindata.to_csv('../Data/Processed/test.csv', encoding='utf-8')

"""## Check Tweets Number Distribution of Different Users"""
random_labels,random_tweets = map(list,zip(*data))
all_text = random_tweets
all_labels = random_labels
train_text = all_text[:train_size]
train_label = all_labels[:train_size]
test_text = all_text[train_size:]
test_label = all_labels[train_size:]

user_frequency = Counter(all_labels)
print(user_frequency)

user_tweets_num = list(user_frequency.values())
print(user_tweets_num)

freq = sorted(user_tweets_num)

## Drawing Distribution Barchart and Boxplot for Tweets Number and Users Number
plt.boxplot(x=freq,whis=1.5,vert=False,showmeans=True)
plt.show()

q1 = np.percentile(freq,25)
q3 = np.percentile(freq,75)
q1 - 1.5*(q3-q1)

median = np.median(freq)
print('median;',median)
print('q1:',q1)
print('q3:',q3)
plt.bar(range(len(freq)), freq)
plt.hlines(median, 0, len(freq),color="red")
plt.xlabel("Number of Users")
plt.ylabel("Number of Tweets")
plt.show()

print(pd.DataFrame(user_tweets_num).describe())

"""## Check the Distribution of Training Set"""
tuser_frequency = Counter(train_label)
tuser_tweets_num = list(tuser_frequency.values())
tfreq = sorted(tuser_tweets_num)

plt.boxplot(x=tfreq,whis=1.5,vert=False,showmeans=True)
plt.show()

tmedian = np.median(tfreq)
plt.bar(range(len(tfreq)), tfreq)
plt.hlines(tmedian, 0, len(tfreq),color="red")
plt.xlabel("Number of Users")
plt.ylabel("Number of Tweets")
plt.show()

print(pd.DataFrame(tuser_tweets_num).describe())