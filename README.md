# Electricity Cost Prediction System (₹)

## Overview
This Streamlit application predicts electricity costs in Indian Rupees (₹) based on various building and environmental factors. The system uses a machine learning model (Random Forest Regressor) trained on historical electricity cost data to provide accurate predictions.

## Features
- **User-friendly interface** with interactive input controls
- **Real-time predictions** displayed in Indian Rupees (₹)
- **Data visualization** including cost distribution and correlation analysis
- **Model information** explaining the prediction methodology

## Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/electricity-cost-predictor.git
   cd electricity-cost-predictor
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the pre-trained model files or train your own:
   - `electricity_cost_model.pkl` (model)
   - `scaler.pkl` (feature scaler)
   - `label_encoder.pkl` (category encoder)

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. The application will open in your default web browser at `http://localhost:8501`

3. Adjust the input parameters in the sidebar and click "Predict Electricity Cost" to see the prediction in ₹

## Input Parameters
- Site Area (sq ft)
- Structure Type (Residential/Commercial/Industrial/Mixed-use)
- Water Consumption (gallons)
- Recycling Rate (%)
- Utilization Rate (%)
- Air Quality Index
- Issue Resolution Time (hours)
- Resident Count

## Output
The system displays:
- Predicted electricity cost in ₹ (Indian Rupees)
- Visualization of historical cost distribution
- Correlation heatmap of factors

## Technical Details
- **Model**: Random Forest Regressor
- **Preprocessing**: Standard Scaling for numerical features, Label Encoding for categorical features
- **Data**: Uses electricity cost dataset with building and environmental metrics

## Screenshots
![App Screenshot](screenshot.png)

## License
MIT License

## Contact
For questions or support, please contact:
[Your Name] - [your.email@example.com]  
Project Link: [https://github.com/yourusername/electricity-cost-predictor](https://github.com/yourusername/electricity-cost-predictor)

---

To create a `requirements.txt` file for this project, include:
```
streamlit==1.12.2
pandas==1.5.3
numpy==1.24.1
scikit-learn==1.2.0
joblib==1.2.0
Pillow==9.4.0
```