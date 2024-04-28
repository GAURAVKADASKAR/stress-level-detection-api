from home.models import *
from rest_framework.decorators import *
from rest_framework import serializers



class InputDataSerializer(serializers.ModelSerializer):
    class Meta:
        model=informationstrees
        fields="__all__"



class PredictionSerializer(serializers.Serializer):
    text = serializers.CharField(max_length=500)