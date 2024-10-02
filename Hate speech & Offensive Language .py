import numpy as np
import pandas as pd
df_twitter=pd.read_csv("/home/karthik/Desktop/internship/Twitter_analysis/train.csv")
df_twitter.tail()
import seaborn as sns
sns.countplot('label',data=df_twitter)
df_twitter.drop('id',axis=1,inplace=True)
df_offensive =pd.read_csv("/home/karthik/Desktop/internship/labeled_data.csv")
df_offensive.tail()
df_offensive.drop(['Unnamed: 0','count','hate_speech','offensive_language','neither'],axis=1,inplace=True)
df_offensive['class'].unique()
sns.countplot('class',data = df_offensive)
df_offensive[df_offensive['class']==0]['class']=1
df_offensive['class'].unique()
df_offensive[df_offensive['class']==0]
df_offensive["class"].replace({0: 1}, inplace=True)
df_offensive['class'].unique()
sns.countplot('class',data=df_offensive)
df_offensive[df_offensive['class']==0]
df_offensive["class"].replace({2: 0}, inplace=True)
sns.countplot('class',data=df_offensive)
df_offensive.rename(columns ={'class':'label'}, inplace = True)
df_offensive.iloc[0]['tweet']
df_offensive.iloc[5]['tweet']
frame=[df_twitter,df_offensive]
df = pd.concat(frame)
sns.countplot('label',data=df)
import nltk
nltk.download('stopwords')
import re
import nltk 
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string
stop_words = set(stopwords.words('english'))
def clean_text(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word.lower()not in stop_words]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text
df['tweet']=df['tweet'].apply(clean_text)
df.to_csv('test.csv',index = False)
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
def make_wordcloud(df):
    comment_words=""
    for val in df.tweet: 
        val = str(val).lower()
        comment_words += " ".join(val)+" "
    print(comment_words[0:100])
    wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = stopwords,min_font_size = 10).generate(comment_words)
  
    plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)

    plt.show()
df.to_csv('testing.csv',index=False)
x = df["tweet"]
y = df["label"]
type(x)
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x, y, random_state=42)
from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer(stop_words = 'english', ngram_range=(1,5))
xtrain_vectorizer = count.fit_transform(xtrain)
xtest_vectorizer = count.transform(xtest)
from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer()
xtrain_tfidf = tfidf.fit_transform(xtrain_vectorizer)
xtest_tfidf = tfidf.transform(xtest_vectorizer)
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
model_vectorizer= MultinomialNB().fit(xtrain_vectorizer, ytrain)
prediction_vectorizer=model_vectorizer.predict(xtest_vectorizer)
print(confusion_matrix(ytest,prediction_vectorizer))
print (classification_report(ytest, prediction_vectorizer))
model_tfidf = MultinomialNB().fit(xtrain_tfidf, ytrain)
prediction_tfidf = model_tfidf.predict(xtest_tfidf)
print (classification_report(ytest, prediction_tfidf))
print(confusion_matrix(ytest,prediction_tfidf))
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding,SpatialDropout1D
from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.models import Sequential
max_words = 50000
max_len = 500
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(xtrain)
sequences = tokenizer.texts_to_sequences(xtrain)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)
model = Sequential()
model.add(Embedding(max_words, 100, input_length=max_len))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
from keras.callbacks import EarlyStopping,ModelCheckpoint
stop = EarlyStopping(monitor='val_accuracy',
                     mode='max',
                     patience=5)
checkpoint= ModelCheckpoint(filepath='./',
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)
test_sequences = tokenizer.texts_to_sequences(xtest)
test_sequences_metrix = sequence.pad_sequences(test_sequences,maxlen = max_len)
accuracy = model.evaluate(test_sequences_metrix,ytest)
lstm_prediction=model.predict(test_sequences_metrix)
res = []
for prediction in lstm_prediction:
    if prediction[0]<0.5:
        res.append(0)
    else:
        res.append(1)
print(confusion_matrix(ytest,res))
import pickle
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol = pickle.HIGHEST_PROTOCOL)
model.save('Hate & Offense_model.h5')
import keras
load_model = keras.models.load_model('./Hate & Offense_model.h5')
with open('tokenizer.pickle', 'rb') as handle:
    load_tokenizer = pickle.load(handle)
test = 'you are beautiful'
test=[clean_text(test)]
print(test)
seq = load_tokenizer.texts_to_sequences(test)
padded = sequence.pad_sequences(seq, maxlen=300)
print(seq)
pred = load_model.predict(padded)
print("pred", pred)
if pred<0.5:
    print("No Hate")
else:
    print("Hate & Offensive")

