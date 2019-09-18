## Epoch 1/5 - 357s - loss: 8.3999 - acc: 0.0112 
## Epoch 2/5 - 350s - loss: 7.7809 - acc: 0.0285 
## Epoch 3/5 - 351s - loss: 7.4458 - acc: 0.0410 
## Epoch 4/5 - 349s - loss: 7.1828 - acc: 0.0511 
## Epoch 5/5 - 349s - loss: 6.9572 - acc: 0.0583 
## Loss: 7.81          Validation Accuracy: 0.05

from pathlib import Path
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import Dense, LSTM, Embedding
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.preprocessing.sequence import pad_sequences

data_folder = Path("../Data/Processed/")
train_file = data_folder / "all_clean_data.csv"

df = pd.read_csv(train_file, encoding='UTF-8')
df = df.fillna(value='')
df['user']=df['user'].apply(str)

tweet_list = df['tweet'].tolist()
user_list = df['user'].tolist()

tokenizer = Tokenizer(num_words=140, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True,split=' ')

tokenizer.fit_on_texts(tweet_list)
X = tokenizer.texts_to_sequences(tweet_list)  # Get the word index
X = pad_sequences(X)

batch_size = 32
model = Sequential()
model.add(Embedding(2000, output_dim = 128, input_length = X.shape[1]))
model.add(LSTM(units = 196, dropout = 0.2, recurrent_dropout = 0.2))
model.add(Dense(9297, activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
print(model.summary())
Y = pd.get_dummies(user_list)
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 0.3, random_state = 42)
model.fit(X_train, Y_train, batch_size = batch_size, epochs = 1, verbose = 2)
loss,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
print("Loss: %.2f" % (loss))
print("Validation Accuracy: %.2f" % (acc))

tokenizer = Tokenizer(num_words=140, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True,split=' ')

tokenizer.fit_on_texts(tweet_list)
X = tokenizer.texts_to_sequences(tweet_list)  # Get the word index
X = pad_sequences(X)

batch_size = 32
model = Sequential()
model.add(Embedding(140, output_dim = 128, input_length = X.shape[1]))
model.add(LSTM(units = 196, dropout = 0.2, recurrent_dropout = 0.2))
model.add(Dense(9297, activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
print(model.summary())
Y = pd.get_dummies(user_list)
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 0.3, random_state = 42)
model.fit(X_train, Y_train, batch_size = batch_size, epochs = 5, verbose = 2)
loss,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
print("Loss: %.2f" % (loss))
print("Validation Accuracy: %.2f" % (acc))