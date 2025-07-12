import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

# Load the trained model and preprocessing objects
model = joblib.load('electricity_cost_model.pkl')
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')

# Structure type mapping - updated to match encoded values
structure_types = {
    'Residential': 0,
    'Commercial': 1,
    'Industrial': 2,
    'Mixed-use': 3
}

def predict_electricity_cost(input_data):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Encode structure type - use correct column name with space
    input_df['structure type'] = le.transform([input_data['structure type']])
    
    # Ensure column order matches training data
    columns_order = [
        'site area', 'structure type', 'water consumption', 
        'recycling rate', 'utilisation rate', 'air qality index',
        'issue reolution time', 'resident count'
    ]
    input_df = input_df[columns_order]
    
    # Scale the features
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    
    return prediction[0]

def main():
    st.title("Electricity Cost Prediction System (₹)")
    st.write("""
    This app predicts the electricity cost in Indian Rupees (₹) based on various building and environmental factors.
    """)
    
    # Sidebar for user input
    st.sidebar.header('User Input Parameters')
    
    def user_input_features():
        site_area = st.sidebar.number_input('Site Area (sq ft)', min_value=500, max_value=10000, value=2000)
        structure_type = st.sidebar.selectbox('Structure Type', list(structure_types.keys()))
        water_consumption = st.sidebar.number_input('Water Consumption (gallons)', min_value=1000, max_value=15000, value=3000)
        recycling_rate = st.sidebar.slider('Recycling Rate (%)', 0, 100, 50)
        utilisation_rate = st.sidebar.slider('Utilization Rate (%)', 0, 100, 70)
        air_quality_index = st.sidebar.number_input('Air Quality Index', min_value=0, max_value=200, value=100)
        issue_resolution_time = st.sidebar.number_input('Issue Resolution Time (hours)', min_value=0, max_value=72, value=24)
        resident_count = st.sidebar.number_input('Resident Count', min_value=0, max_value=500, value=100)
        
        data = {
            'site area': site_area,
            'structure type': structure_type,
            'water consumption': water_consumption,
            'recycling rate': recycling_rate,
            'utilisation rate': utilisation_rate,
            'air qality index': air_quality_index,
            'issue reolution time': issue_resolution_time,
            'resident count': resident_count
        }
        
        return data
    
    input_data = user_input_features()
    
    # Display user input
    st.subheader('User Input Parameters')
    st.write(pd.DataFrame(input_data, index=[0]))
    
    # Prediction (now in ₹)
    if st.button('Predict Electricity Cost'):
        prediction = predict_electricity_cost(input_data)
        st.subheader('Prediction')
        st.success(f'Predicted Electricity Cost: ₹{prediction:,.2f}')  # Changed to ₹ and added comma formatting
    
    # Data visualization section
    st.subheader('Data Analysis (₹)')
    
    # Load sample data for visualization
    @st.cache
    def load_data():
        return pd.read_csv(r"C:\Users\Lingala Anusha\OneDrive\Desktop\electricity_cost_dataset.csv")
    
    df = load_data()
    
    if st.checkbox('Show Raw Data'):
        st.subheader('Raw Data')
        st.write(df)
    
    # Show distribution of electricity cost (now in ₹)
    st.subheader('Electricity Cost Distribution (₹)')
    st.line_chart(df['electricity cost'])
    
    # Correlation heatmap
    if st.checkbox('Show Correlation Heatmap'):
        st.subheader('Correlation Heatmap')
        numeric_df = df.select_dtypes(include=[np.number])
        st.write(numeric_df.corr())
    
    # Model information
    st.subheader('Model Information')
    st.write("""
    This prediction uses a Random Forest Regressor model trained on historical electricity cost data.
    The model considers various factors including building characteristics, usage patterns, and environmental factors.
    All cost predictions are displayed in Indian Rupees (₹).
    """)

if __name__ == '__main__':
    main()