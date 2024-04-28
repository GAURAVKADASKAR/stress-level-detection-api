from flask import Flask, request, render_template, jsonify
import nltk
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Disable warnings
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Read the file
def read_file(file_path):
    with open(file_path, 'r', errors='ignore') as file:
        raw_text = file.read().lower()
    return raw_text

# Preprocess the text
def preprocess_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [token.lower() for token in tokens]
    lemmatizer = nltk.stem.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

# Greetings
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey")
GREETING_RESPONSES = ["hi", "hey", "nods", "hi there", "hello", "I am glad! you are talking to me"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

# Response generation
def generate_response(user_response, sent_tokens):
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=nltk.word_tokenize, stop_words="english")
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 1:
        return "I am sorry! I don't understand you"
    else:
        return sent_tokens[idx]

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.form['user_input']
    raw_text = read_file('demo (1).txt')
    sent_tokens = nltk.sent_tokenize(raw_text)
    if user_input.lower() == "bye":
        response = "miya: Bye! Have a great time!"
    else:
        if greeting(user_input) is not None:
            response = "Aneka: " + greeting(user_input)
        else:
            response = generate_response(user_input.lower(), sent_tokens)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
