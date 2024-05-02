from flask import Flask, request, render_template
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
import contractions
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from collections import Counter
import pickle

clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

app = Flask(__name__)

emoji_pattern = re.compile('(?::|;|=)(?:-)?(?:\)|\(|D|P)')
nltk.download('punkt')
nltk.download('wordnet')


def cleaning_the_data(text):
    if not isinstance(text, str):
        text = str(text)

    new_text = re.sub(r"'s\b'", ' is', text)
    new_text = re.sub(r"#", " ", new_text)
    new_text = re.sub(r"@[A-Za-z0-9]+", " ", new_text)
    new_text = re.sub(r"http\s+", " ", new_text)
    new_text = contractions.fix(new_text)
    new_text = re.sub(r"[^a-zA-Z]", " ", new_text)
    new_text = re.sub(emoji_pattern, '', new_text)
    new_text = new_text.lower().strip()
    return new_text


def normalize_text(text):
    cleaned_text = cleaning_the_data(text)
    # Tokenize the text
    tokens = word_tokenize(cleaned_text)

    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word, pos='a') for word in tokens if len(word) > 2]

    # Join tokens back into a single string
    normalized_text = ' '.join(lemmatized_tokens)
    return normalized_text


# train_data['normalized_text'] = train_data['text'].apply(normalize_text)


@app.route('/',methods=['POST','GET'])
def index():
    return render_tmplate('index.html')

@app.route('/predict', methods=['POST','GET'])
def index():
    if(request.method=='POST'):
        message = request.form['message']
        cleaned_message = normalize_text(message)
        message_vector = tfidf.transform([cleaned_message])
        prediction = clf.predict(message_vector)[0]

        return render_tmplate('index.html', predition=prediction)


if __name__ == '__main__':
    app.run(debug=True)


