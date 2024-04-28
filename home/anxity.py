# prediction_app/views.py
from django.http import JsonResponse
import re
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required resources from NLTK
nltk.download('stopwords')
nltk.download('punkt')

# Preprocessing functions
def remove_url(text):
    re_url = re.compile('https?://\S+|www\.\S+')
    return re_url.sub('', text)

def remove_punctuation(text):
    exclude = string.punctuation
    return text.translate(str.maketrans('', '', exclude))

def remove_stopwords(text):
    stopwords_set = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word not in stopwords_set]
    return ' '.join(filtered_words)

def perform_stemming(text):
    stemmer = PorterStemmer()
    words = word_tokenize(text)
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)

# Load pre-trained model and vectorizer
loaded_model = pickle.load(open('anxiety_trained2.sav', 'rb'))
vectorizer = pickle.load(open('vectorizer1.sav', 'rb'))

def predict(request):
    if request.method == 'POST':
        data = request.POST
        if 'text' not in data:
            return JsonResponse({'error': 'Text data is missing'}, status=400)
        
        text = data['text']
        
        # Perform preprocessing steps
        text = remove_url(text)
        text = remove_punctuation(text)
        text = remove_stopwords(text)
        text = perform_stemming(text)
        
        # Transform the test data using the vectorizer
        X_test_array = vectorizer.transform([text]).toarray()
        
        # Make a prediction using the loaded model
        prediction = loaded_model.predict(X_test_array)
        
        # Return the prediction as a JSON response
        return JsonResponse({'prediction': prediction[0]})
