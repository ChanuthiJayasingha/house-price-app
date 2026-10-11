import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings('ignore')

st.title("XGBoost House Price Predictor")
st.write("Upload a CSV matching the 75 Ames Housing features. The app applies the full preprocessing pipeline and returns the estimated sale prices.")

@st.cache_resource
def load_models():
    pipeline = joblib.load('xgb_housing_pipeline.joblib')
    expected_cols = joblib.load('feature_columns.joblib')
    return pipeline, expected_cols

try:
    pipeline, expected_cols = load_models()
except Exception as e:
    st.error(f"Error loading models: {str(e)}. Ensure your .joblib files are uploaded.")
    st.stop()

uploaded_file = st.file_uploader("Upload CSV with Housing Features", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Enforce exact data types from training
        for col in ["MSSubClass", "MoSold"]:
            if col in df.columns:
                df[col] = df[col].astype(str)
                
        # Schema validation
        missing_cols = set(expected_cols) - set(df.columns)
        if missing_cols:
            st.error(f"Schema Error: Missing required columns: {missing_cols}")
        else:
            X_input = df[expected_cols]
            log_predictions = pipeline.predict(X_input)
            actual_predictions = np.expm1(log_predictions)
            
            df['Predicted_SalePrice'] = actual_predictions
            st.success("Success! Predictions generated.")
            
            # Convert dataframe to CSV for download
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name='predicted_house_prices.csv',
                mime='text/csv',
            )
            
    except Exception as e:
        st.error(f"Runtime Error: {str(e)}")