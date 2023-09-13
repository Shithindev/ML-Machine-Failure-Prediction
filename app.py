import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the saved model
model = joblib.load(r'D:\ML_Project\rfc_model.pkl')

st.title('Machine Failure Prediction')

# Input widgets
air_temp = st.number_input(str('Enter Air temperature [K]'), min_value=0.0, max_value=1000.0, value=300.6)
process_temp = st.number_input(str('Enter Process temperature [K]'), min_value=0.0, max_value=1000.0, value=309.0)
rotational_speed = st.number_input(str('Enter Rotational speed [rpm]'), min_value=0.0, max_value=5000.0, value=1596.0)
torque = st.number_input(str('Enter Torque [Nm]'), min_value=0.0, max_value=100.0, value=36.1)
tool_wear = st.number_input(str('Enter Tool wear [min]'), min_value=0.0, max_value=1000.0, value=140.0)
type_input = st.selectbox(str('Select Type'), options=['Heavy', 'Light', 'Medium'])

# Input widgets for missing features
twf = st.checkbox('TWF (Tool Wear Failure)')
hdf = st.checkbox('HDF (Heat-Related Failure)')
osf = st.checkbox('OSF (Overload-Related Failure)')

# Load the StandardScaler object using a relative path
try:
    ss = joblib.load(r'D:\ML_Project\standard_scaler.pkl')  # Use a relative path if files are in the same directory
except FileNotFoundError:
    st.error("StandardScaler file not found. Please check the file path.")
    st.stop()

# Make predictions
if st.button('Predict'):
    # Prepare the input data for prediction
    input_data = np.array([[air_temp, process_temp, rotational_speed, torque, tool_wear]])
    
    # Perform one-hot encoding for 'type_input'
    type_encoded = pd.get_dummies(pd.DataFrame({'Type': [type_input]}))
    
    # Add the missing features to the input data
    input_data = np.hstack([input_data, type_encoded.values, [[twf, hdf, osf]]])
    
    # Standardize the input data using the loaded scaler
    input_data = ss.transform(input_data)

    # Make predictions
    prediction = model.predict(input_data)[0]

    # Display the prediction
    if prediction == 1:
        st.success('FAILURE!')
    else:
        st.error('NON-FAILURE')





