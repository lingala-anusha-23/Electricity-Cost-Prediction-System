import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

def load_and_preprocess_data(filepath):
    # Load the dataset
    df = pd.read_csv(filepath)
    
    # Handle missing values if any
    df = df.dropna()
    
    # Check column names
    print("Columns in dataset:", df.columns.tolist())
    
    # Encode categorical variables - use correct column name with space
    le = LabelEncoder()
    df['structure type'] = le.fit_transform(df['structure type'])
    
    # Feature selection - using all available features
    X = df.drop('electricity cost', axis=1)
    y = df['electricity cost']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, le

def train_models(X_train, y_train):
    # Initialize models
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Linear Regression': LinearRegression()
    }
    
    # Train models
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    return trained_models

def evaluate_models(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {'MSE': mse, 'R2': r2}
    return results

def save_models_and_scalers(models, scaler, le):
    # Save the best model
    joblib.dump(models['Random Forest'], 'electricity_cost_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(le, 'label_encoder.pkl')

def main():
    # File path to your dataset
    filepath =(r"C:\Users\Lingala Anusha\OneDrive\Desktop\electricity_cost_dataset.csv")
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, scaler, le = load_and_preprocess_data(filepath)
    
    # Train models
    models = train_models(X_train, y_train)
    
    # Evaluate models
    results = evaluate_models(models, X_test, y_test)
    for name, metrics in results.items():
        print(f"{name}: MSE = {metrics['MSE']:.2f}, R2 = {metrics['R2']:.2f}")
    
    # Save the best model and preprocessing objects
    save_models_and_scalers(models, scaler, le)

if __name__ == "__main__":
    main()