import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
import os

# Set page config
st.set_page_config(page_title="Electricity Cost Prediction", layout="wide")

# Structure type mapping
structure_types = {
    'Residential': 0,
    'Commercial': 1,
    'Industrial': 2,
    'Mixed-use': 3
}

# Load models with cache_resource (for objects that shouldn't be copied)
@st.cache_resource
def load_models():
    try:
        return (
            joblib.load('electricity_cost_model.pkl'),
            joblib.load('scaler.pkl'),
            joblib.load('label_encoder.pkl')
        )
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        st.info("Please ensure you've run the training script first ('Electricity Cost Prediction.py')")
        st.stop()

# Load models
model, scaler, le = load_models()

def predict_electricity_cost(input_data):
    input_df = pd.DataFrame([input_data])
    input_df['structure type'] = le.transform([input_data['structure type']])
    columns_order = [
        'site area', 'structure type', 'water consumption', 
        'recycling rate', 'utilisation rate', 'air qality index',
        'issue reolution time', 'resident count'
    ]
    input_df = input_df[columns_order]
    input_scaled = scaler.transform(input_df)
    return model.predict(input_scaled)[0]

# Load data with cache_data (for dataframes)
@st.cache_data
def load_data():
    try:
        return pd.read_csv(r"C:\Users\Lingala Anusha\OneDrive\Desktop\AICTE INTERNSHIPS\Aicte Microsoft AI Azure Internship\Electricity Cost Prediction System\electricity_cost_dataset.csv")
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

@st.cache_data
def get_model_predictions(_df):
    """Generate predictions for visualization"""
    try:
        df = _df.copy().dropna()
        df['structure type'] = le.transform(df['structure type'])
        
        X = df.drop('electricity cost', axis=1)
        y = df['electricity cost']
        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)
        
        return y, y_pred
    except Exception as e:
        st.error(f"Error generating predictions: {str(e)}")
        return None, None

def main():
    st.title("Electricity Cost Prediction System (₹)")
    st.write("This app predicts electricity costs based on building and environmental factors.")
    
    # User input section
    st.sidebar.header('User Input Parameters')
    def user_input_features():
        return {
            'site area': st.sidebar.number_input('Site Area (sq ft)', 500, 10000, 2000),
            'structure type': st.sidebar.selectbox('Structure Type', list(structure_types.keys())),
            'water consumption': st.sidebar.number_input('Water Consumption (gallons)', 1000, 15000, 3000),
            'recycling rate': st.sidebar.slider('Recycling Rate (%)', 0, 100, 50),
            'utilisation rate': st.sidebar.slider('Utilization Rate (%)', 0, 100, 70),
            'air qality index': st.sidebar.number_input('Air Quality Index', 0, 200, 100),
            'issue reolution time': st.sidebar.number_input('Issue Resolution Time (hours)', 0, 72, 24),
            'resident count': st.sidebar.number_input('Resident Count', 0, 500, 100)
        }
    
    input_data = user_input_features()
    st.subheader('User Input Parameters')
    st.write(pd.DataFrame(input_data, index=[0]))
    
    if st.button('Predict Electricity Cost'):
        prediction = predict_electricity_cost(input_data)
        st.success(f'Predicted Electricity Cost: ₹{prediction:,.2f}')

    # Load data
    df = load_data()
    if df is None:
        st.stop()
    
    # Visualization section
    st.subheader('Model Performance Analysis')
    
    # Load data and predictions
    y_actual, y_predicted = get_model_predictions(df)
    
    if y_actual is not None:
        # Actual vs Predicted plot
        fig1, ax1 = plt.subplots(figsize=(10,6))
        ax1.scatter(y_actual, y_predicted, alpha=0.6)
        ax1.plot([y_actual.min(), y_actual.max()], 
                 [y_actual.min(), y_actual.max()], 'k--', lw=2)
        ax1.set_xlabel('Actual Cost (₹)')
        ax1.set_ylabel('Predicted Cost (₹)')
        ax1.set_title('Actual vs Predicted Costs')
        st.pyplot(fig1)
        
        # Metrics
        st.write(f"""
        **Model Performance Metrics:**
        - R² Score: {r2_score(y_actual, y_predicted):.3f}
        - Mean Absolute Error: ₹{mean_absolute_error(y_actual, y_predicted):.2f}
        """)
        
        # Error distribution
        fig2, ax2 = plt.subplots(figsize=(10,6))
        errors = y_actual - y_predicted
        ax2.hist(errors, bins=30)
        ax2.set_xlabel('Prediction Error (₹)')
        ax2.set_title('Error Distribution')
        st.pyplot(fig2)
    
    # Data exploration
    st.subheader('Data Exploration')
    if st.checkbox('Show Raw Data'):
        st.write(df)
    
    st.line_chart(df['electricity cost'])
    
    if st.checkbox('Show Correlation Heatmap'):
        numeric_df = df.select_dtypes(include=[np.number])
        st.write(numeric_df.corr())

if __name__ == '__main__':
    main()