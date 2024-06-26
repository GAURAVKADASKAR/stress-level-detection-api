from django.contrib import admin
from django.urls import path,include
from home.models import *
from home.views import *

urlpatterns = [
    path('predict/',PredictStressLevel.as_view()),
    path('anxi/', PredictAPIView.as_view(), name='predict'),
    path('sleep/',PredictSleepQuality.as_view()),
    path('chatbot/', ChatbotView.as_view(), name='chatbot'),
    path('admin/', admin.site.urls),
]