from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .my_project import MyModel

class PredictStressLevel(APIView):
    def post(self, request):
        data = request.data
        model = MyModel()
        prediction = model.stresslevel_prediction([data.get('humidity', 0), data.get('Temperature', 0), data.get('step_count', 0)])
        return Response({'prediction': prediction}, status=status.HTTP_200_OK)

# prediction_app/views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializer import PredictionSerializer
import re
import nltk
from random import randint
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

class PredictAPIView(APIView):
    def post(self, request):
        serializer = PredictionSerializer(data=request.data)
        if serializer.is_valid():
            text = serializer.validated_data['text']
            
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
            if(prediction==0):
                happy = ["Take a moment to reflect on the things you're grateful for in your life.",
"Get moving and enjoy some physical activity.",
"Reach out to friends or family members and spend quality time with them. ",
"Engage in a creative activity that brings you joy, such as painting, writing, or playing music. ",
"Practice mindfulness and savor the present moment. ",
" Dedicate time to a hobby or passion that you enjoy." ,
" Pamper yourself and indulge in a little self-care.",
"Spend time outdoors and connect with nature. ",
" Spread joy and positivity by performing random acts of kindness for others.",
"Set realistic goals for yourself and take steps towards achieving them."
]
                return Response({'prediction':happy[randint(0,9)]})
            else:
                array = ["Take a few minutes each day to practice mindfulness or meditation. ", 
" Engage in regular physical activity, whether it's going for a walk, practicing yoga, or participating in your favorite sport",
"Reach out to friends or family members for support. ",
"Take breaks from social media and news outlets, especially if they contribute to your anxiety.",
"Take time each day to reflect on things you're grateful for.",
"Spend time doing activities that you enjoy and that bring you pleasure. ",
"Prioritize getting enough sleep each night. Lack of sleep can exacerbate feelings of anxiety and make it harder to cope with stress.",
"Consider reaching out to a mental health professional for support. ",
"Be kind to yourself and practice self-compassion. ",
"Instead of dwelling on things that are beyond your control, focus on the aspects of your life that you can influence.",
" Incorporate deep breathing exercises into your daily routine.",
"Reduce your intake of caffeine and alcohol, as they can exacerbate feelings of anxiety and interfere with sleep. ",
" Break tasks down into smaller, manageable goals. ",
"Take a walk in nature or spend time outdoors.",
"Watch a funny movie, read a humorous book, or spend time with people who make you laugh.",
"Prioritize self-care activities that nurture your mind, body, and soul. ",
" Experiment with relaxation techniques such as progressive muscle relaxation, guided imagery, or aromatherapy. ",
"Helping others can provide a sense of purpose and fulfillment. ",
"Accept that perfection is unattainable and embrace imperfection. ",
" Engage in creative activities such as writing, drawing, or crafting. ",
"Progressive muscle relaxation involves tensing and then relaxing each muscle group in your body, one at a time. ",
"Repeat positive affirmations to yourself regularly. ",
"Designate a quiet, comfortable space in your home where you can go to relax and unwind.",
" Avoid using electronic devices such as smartphones or computers before bedtime, as the blue light emitted from screens can disrupt sleep patterns. ",
" Create a daily routine that includes regular mealtimes, exercise, and relaxation. ",
"Grounding techniques can help bring you back to the present moment and reduce feelings of anxiety. ",
"Start a gratitude jar where you can write down things you're grateful for each day. ",
"Listen to calming or uplifting music that resonates with you.",
"Close your eyes and visualize yourself in a peaceful, calming environment.",
"Consider joining a support group or online community for individuals experiencing anxiety. "
]
            return Response({'prediction': array[randint(0,29)]}, status=status.HTTP_200_OK)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    

import pickle
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

class PredictSleepQuality(APIView):
    def post(self, request):
        # Load the trained model
        with open('sleep1.sav', 'rb') as f:
            loaded_model = pickle.load(f)
        
        # Extract input data from the request
        data = request.data
        sleep_duration = data.get('sleep_duration', 0)
        efficiency = data.get('efficiency', 0)
        disturbance = data.get('disturbance', 0)

        # Organize input data into the format expected by the model
        X_test = [[sleep_duration, efficiency, disturbance]]

        # Make predictions using the loaded model
        prediction = loaded_model.predict(X_test)

        # Return the prediction as a JSON response
        return Response({'prediction': prediction}, status=status.HTTP_200_OK)
