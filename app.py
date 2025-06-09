import tensorflow as tf
import streamlit as st
import pickle
import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler,LabelEncoder
from tensorflow.keras.models import load_model


# load the trained model scaler pickel,onehot
model=load_model('model.h5')

with open('OneHotEncoder_geo.pkl','rb') as file:
    onehotencoder_geo=pickle.load(file)
with open('Label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)
with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

#Streamlit app

st.title("Customer Churn Prediction")

#User input
geography=st.selectbox('Geography',onehotencoder_geo.categories_[0])
gender=st.selectbox('Gender',label_encoder_gender.classes_)
age=st.slider('Age',18,92)
balance= st.number_input("Balance")
credit_score= st.number_input("Credit Score")
estimate_salary= st.number_input("Estimated salary")
tenure=st.slider("Tenure", 0,10)
num_of_products=st.slider("Number of products",1,4)
has_cr_card=st.selectbox("Has Credit card",[0,1])
is_active_member=st.selectbox("is active member",[0,1])

input_data=pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender' :[label_encoder_gender.transform([gender])[0]],
    'Age' :[age],
    'Tenure' :[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimate_salary]

 })

geo_encoded=onehotencoder_geo.transform([[geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=onehotencoder_geo.get_feature_names_out(['Geography']))

input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)
input_data_scaled=scaler.transform(input_data)

prediction=model.predict(input_data_scaled)
prediction_proba=prediction[0][0]

st.write(f'Churn Probability :{prediction_proba:.2f}')
if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn')