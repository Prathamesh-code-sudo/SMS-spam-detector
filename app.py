import streamlit as st 
import pickle
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

# load model and vectorizer
model = pickle.load(open('model.pkl','rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

ps = PorterStemmer()

def preprocess(text):
    text = re.sub(r'[^a-zA-Z]',' ',text)
    text = text.lower()
    words = text.split()
    stop_words = stopwords.words('english')
    words = [ps.stem(w) for w in words if w not in stop_words]
    return " ".join(words)

#UI

st.set_page_config(page_title="SMS Spam Detector")

st.title("SMS Detection App")
st.write("Detect Whether a message is Spam or Ham")

#input
message = st.text_area("Enter your message: ")

if st.button("check"):
    if message.strip() != "":
        cleaned = preprocess(message)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        if prediction == 1:
            st.error("Spam Message")
        else:
            st.success("Not Spam")
    
    else:
        st.warning("Please enter a message: ")
