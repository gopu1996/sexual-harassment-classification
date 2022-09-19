import streamlit as st
import joblib
import json
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
model_tfidf = joblib.load('sexual-harasment-tf-idf')
model = joblib.load('sexual-harasment-model')


def fn_preprocess_text(sentence):
    stop_words1 = set(stopwords.words('english'))
    stop_words2 = set('and the was to in me my of at it when were by this\
    with that from there one for is we not so are then day had all'.split())
    stop_words = stop_words1 | stop_words2

    stemmer = PorterStemmer()
    text = str(sentence).lower()

    text = text.replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")
    text = text.replace("what's", "what is").replace("it's", "it is").replace("i'm", "i am")
    text = text.replace("he's", "he is").replace("she's", "she is").replace("'s", " own")
    text = text.replace("'ll", " will").replace("n't", " not").replace("'re", " are").replace("'ve", " have")
    text = text.replace("?", "").replace("i'm", " i am").replace("what's", " what is")

    text = re.sub('[^a-zA-Z0-9\n]', ' ', text)  # ------------------- Replace every special char with space
    text = re.sub('\s+', ' ', text).strip()  # ---------------------- Replace excess whitespaces

    text = text.split()
    text = [i for i in text if i.lower() not in stop_words]
    singles = [stemmer.stem(plural) for plural in text]
    single = ' '.join(singles)
    return single.lower()



st.title('Classify Sexual Harassment Stories')
txt = st.text_area("Describe Your Stories Here :")



if st.button('Submit'):
    preprocess_txt = fn_preprocess_text(txt)
    input_data_features = model_tfidf.transform([preprocess_txt])
    prediction = model.predict(input_data_features)
    prediction_proba = model.predict_proba(input_data_features)

    col1, col2 = st.columns(2)

    with col1:
        st.write("Prediction : ", prediction[0])
    with col2:
        if prediction[0] == 0:
            st.write("Prediction Probability: ",prediction_proba[0][0])
        else:
            st.write("Prediction Probability: ", prediction_proba[0][1])