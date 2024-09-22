import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the pre-trained model
car_price_model = pickle.load(open('car_price_model.pkl', 'rb'))

# Load the CSV data for encoders (to maintain label mappings)
df = pd.read_csv('car_prediction_data.csv')

# Set up LabelEncoder to transform user inputs
label_encoder_model = LabelEncoder()
label_encoder_city = LabelEncoder()
label_encoder_fuel = LabelEncoder()
label_encoder_ownership = LabelEncoder()
label_encoder_engine = LabelEncoder()

# Fit the encoders with the existing data
df['model'] = label_encoder_model.fit(df['model'])
df['city'] = label_encoder_city.fit(df['city'])
df['fuel_type'] = label_encoder_fuel.fit(df['fuel_type'])
df['ownership'] = label_encoder_ownership.fit(df['ownership'])

# Streamlit App
st.title('Car Price Prediction')

# Input fields for user data
selected_model = st.selectbox('Select Car Model', label_encoder_model.classes_)
city = st.selectbox('Select City', label_encoder_city.classes_)
registration_year = st.slider('Select Car Year', min_value=2000, max_value=2024, step=1)
km = st.number_input('Enter Kilometers Driven', min_value=0)
fuel_type = st.selectbox('Select Fuel Type', label_encoder_fuel.classes_)
transmission = st.radio('Transmission Type', ['Manual', 'Automatic'])
max_power = st.number_input('Enter Max Power (in bhp)', min_value=0)
mileage = st.number_input('Enter Mileage (in kmpl)', min_value=0)
ownership = st.selectbox('Select Ownership', label_encoder_ownership.classes_)
torque = st.number_input('Enter Torque', min_value=0)
seats = st.number_input('Enter Number of Seats', min_value=2, max_value=10)

# Convert transmission to numeric
transmission = 0 if transmission == 'Manual' else 1

# Preprocess user input
input_data = pd.DataFrame({
    'city': [label_encoder_city.transform([city])[0]],
    'model': [label_encoder_model.transform([selected_model])[0]],
    'registration_year': [registration_year],
    'ownership': [label_encoder_ownership.transform([ownership])[0]],  # Add ownership encoding
    'km': [km],
    'fuel_type': [label_encoder_fuel.transform([fuel_type])[0]],
    'transmission': [transmission],
    'mileage': [mileage],
    'max_power': [max_power],
    'torque': [torque],
    'seats': [seats],
})

# Predict button
if st.button('Predict Car Price'):
    prediction = car_price_model.predict(input_data)
    st.success(f'The predicted price for the car is: ₹{prediction[0]:,.2f}')
