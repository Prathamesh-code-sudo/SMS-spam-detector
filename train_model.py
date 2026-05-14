import pandas as pd
import re
import nltk
import pickle

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

nltk.download('stopwords')

# Load Dataset
df  = pd.read_csv(r"C:\Users\prath\Desktop\SMS Spam_detector using ML\spam.csv",encoding = 'latin-1')
df = df[['v1','v2']]
df.columns = ['label','message']
df['label'] = df['label'].map({'ham':0, 'spam':1})

ps = PorterStemmer()

def preprocess(text):
    text = re.sub('[^a-zA-Z]'," ", text)
    text = text.lower()
    words = text.split()
    words = [ps.stem(w) for w in words if w not in stopwords.words('english')]
    return " ".join(words)

df['cleaned'] = df['message'].apply(preprocess)

# vectorization
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['cleaned'])
y = df['label']

# train model
model = MultinomialNB()
model.fit(X,y)

# save model & vectorizer
pickle.dump(model,open('model.pkl','wb'))
pickle.dump(tfidf,open('vectorizer.pkl','wb'))

print("Model saved successfully!!")
