import numpy as np
import pickle
import streamlit as st  
from random import randint

class MyModel:
    def __init__(self):
        self.loaded_model = pickle.load(open(r'C:/Users/Dell/Desktop/djangoworkspace/hack/stress-level-detection.py','rb'))

    def stresslevel_prediction(self, input_data):
        id_np_array = np.asarray(input_data)
        id_reshaped = id_np_array.reshape(1,-1)
        prediction = self.loaded_model.predict(id_reshaped)

        if prediction[0] == 0:
           
            low = [
	" Even when your stress level is low, it's beneficial to continue practicing relaxation techniques such as deep breathing, progressive muscle relaxation, or meditation.",
	"Keep up with healthy habits that contribute to stress reduction, such as getting regular exercise, eating nutritious meals, prioritizing sleep, and staying hydrated. ",
	"Continue to engage in activities that bring you joy and relaxation.",
	"Cultivate an attitude of gratitude by regularly acknowledging and appreciating the positive aspects of your life. ",
	"Be mindful of your limits and set boundaries to protect your time and energy. "
]
                                                    
            return ("-------------------------------  Stress Level : Low ---------------------------  " ,low[randint(0,4)])
        elif prediction[0] == 1:
            
            medium = [
	"Incorporate stress-relief techniques into your daily routine. ",
	" Schedule regular breaks throughout your day to rest and recharge.  ",
	"  Break down your tasks into smaller, more manageable steps, and prioritize them based on importance and urgency. ",
	"  Incorporate regular physical activity into your routine, as exercise is a powerful stress reliever. ",
	" Reach out to friends, family members, or a trusted mentor for support.  "
]
            return ("----------------------------  Stress Level : Medium ---------------------------" , medium[randint(0,4)])
        else:   
          
            heigh = [
	"Dedicate time each day to practice relaxation techniques such as deep breathing, progressive muscle relaxation, or guided imagery.",
	"Assess your commitments and responsibilities, and prioritize tasks based on importance and urgency.",
	" Incorporate activities into your routine that help you relax and unwind. ",
	"Prioritize self-care activities that nourish your physical, mental, and emotional well-being. ",
	"If your stress level remains high and is significantly impacting your daily functioning and quality of life, consider seeking support from a mental health professional. "
]
            data="-------------------------------  Stress Level : High ---------------------------  "
            name=heigh[randint(0,4)]
            return f"{data}\n{name}"
            


                            
def main():
    st.title('STRESS LEVEL PREDICTION WEB APP')
    
    Humidity = st.text_input('Humidity Value')
    Temperature = st.text_input('Body Temperature')
    Step_count = st.text_input('Number of Steps')
    
    model = MyModel()
    
    # Prediction code
    diagnosis = ''
    
    if st.button('PREDICT'):
        diagnosis = model.stresslevel_prediction([Humidity,Temperature,Step_count])
        
    st.success(diagnosis)
                            
if __name__ == '__main__':
    main()
