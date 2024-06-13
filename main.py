import streamlit as st
import pandas as pd
import numpy as np
import joblib


# Load the pre-trained model and encoders

label_encoders = joblib.load('label_encoders.pkl')
xgbrmodel = joblib.load('model1.pkl')
xgbrmodel2 = joblib.load('model2.pkl')

features = pd.read_csv('feature_data.csv')

#function to predict the price
def predict_price(make, model, body, year, engine, fuel, transmission):

    # Create a DataFrame for the input features
    input_df = pd.DataFrame({
        'make': [make],
        'model': [model],
        'body': [body],
        'year': [year],
        'engine': [engine],
        'fuel': [fuel],
        'transmission': [transmission]
    })

    for column, le in label_encoders.items():
        input_df[column] = le.transform(input_df[column].astype(str))
    X = input_df.values  # Convert to numpy array

    # Predict the price using the trained model
    y_pred = xgbrmodel2.predict(X)
    return y_pred[0] / 1000000 


st.title('Used Car Price Pakistan')
st.caption('A simple web app to predict the price of a used car in Pakistan.')

with st.sidebar:
    st.link_button("GitHub", "https://github.com/Shaheer04", use_container_width=True)
    st.link_button("LinkedIn", "https://www.linkedin.com/in/shaheerjamal", use_container_width=True)
    st.info('This car price prediction model provides estimated values based on input features and training data. However, the accuracy of these predictions may be limited due to low data points for some features and other inherent limitations of the model.')

# Create input fields for the user to enter the car details

make = st.selectbox('Make', features['make'].unique(), index=None)
model = st.selectbox('Model', features[features['make'] == make]['model'].unique(),index=None)
body = st.selectbox('Body', features[features['model'] == model]['body'].unique())
year = st.selectbox('year', features[features['model'] == model]['year'].unique().astype(int))
engine = st.selectbox('Engine', features[features['model'] == model]['engine'].unique().astype(int))
fuel = st.selectbox('Fuel', features[features['model'] == model]['fuel'].unique())
transmission = st.selectbox('Transmission', features[features['model'] == model]['transmission'].unique())


if st.button('Predict Price', type='primary'):
    prediction = predict_price(make, model, body, year, engine, fuel, transmission)
    result = f"{prediction:.1f} Million PKR"
    st.success(f'The estimated price of this car is {result}') 
    st.caption('Made with ❤️ by Shaheer Jamal')