from flask import Flask, request, render_template
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import contractions
from nltk.stem import PorterStemmer
from collections import Counter
import pickle
import sklearn
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

clf = pickle.load(open('clf1.pkl', 'rb'))
tfidf = pickle.load(open('tfidf1.pkl', 'rb'))

emoji_pattern = re.compile('(?::|;|=)(?:-)?(?:\)|\(|D|P)')

nltk_stopwords = set(stopwords.words('english'))
nltk_stopwords.remove('not')

sklearn_stopwords = set(ENGLISH_STOP_WORDS)

# joining both the stopwords
all_stopwords = nltk_stopwords.union(sklearn_stopwords)

all_stopwords_list = list(all_stopwords)


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

    new_text = re.sub(r"\b(?:not|no|never|none|nobody|nowhere|nothing)\b[\w\s]+[^\w\s]", "not_negative", new_text)
    return new_text


def normalize_text(text):
    cleaned_text = cleaning_the_data(text)
    # Tokenize the text
    tokens = word_tokenize(cleaned_text)
    
    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word, pos='a') for word in tokens if len(word) > 2]
    
    # Remove stopwords
    filtered_tokens = [word for word in lemmatized_tokens if word.lower() not in all_stopwords_list]
    
    # Join tokens back into a single string
    normalized_text = ' '.join(filtered_tokens)
    return normalized_text

app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template("index.html")


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        cleaned_message = normalize_text(message)
        message_vector = tfidf.transform([cleaned_message])
        prediction = clf.predict(message_vector)
        return render_template("index.html", prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
