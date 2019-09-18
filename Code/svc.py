## LinearSVC 0.30128

from pathlib import Path
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer 
from sklearn.svm import LinearSVC

## Running on local machine
## All Data
data_folder = Path("../Data/Processed/")
train_file = data_folder / "all_clean_data.csv"
test_file = data_folder / "test_clean_data.csv"

traindf = pd.read_csv(train_file, encoding='UTF-8') 
testdf = pd.read_csv(test_file, encoding='UTF-8') 
user_list = traindf['user']
tweet_list = traindf['tweet']
#test_user_list = testdf['user'].tolist()
test_tweet_list = testdf['tweet']

count_v1= CountVectorizer()
counts_train = count_v1.fit_transform(tweet_list.values.astype('U'))
print("the shape of train is "+repr(counts_train.shape))   
count_v2 = CountVectorizer(vocabulary=count_v1.vocabulary_)
counts_test = count_v2.fit_transform(test_tweet_list.values.astype('U'))
print("the shape of test is "+repr(counts_test.shape)) 

tfidftransformer = TfidfTransformer()
X_train = tfidftransformer.fit(counts_train).transform(counts_train)
X_test = tfidftransformer.fit(counts_test).transform(counts_test)
y_train = user_list

clf = LinearSVC(random_state=0, tol=1e-5)
clf.fit(X_train, y_train) 
preds = clf.predict(X_test)
num = 0
preds = preds.tolist()
id_list = list(range(1,35438))
result = pd.DataFrame(list(zip(id_list,preds)),columns =['Id','Predicted'])
result.to_csv('../Result/result_svc.csv',index=False)
