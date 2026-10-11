import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings('ignore')

# 1. UI Configuration (Must be the first Streamlit command)
st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")

st.title("🏠 XGBoost House Price Predictor")
st.markdown("Predict the sale price of houses in Ames, Iowa using an advanced machine learning model.")

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

# 2. Create Tabs for different User Experiences
tab1, tab2 = st.tabs(["📂 Bulk Predict (CSV Upload)", "✍️ Single House (Manual Entry)"])

# ---------------------------------------------------------
# TAB 1: CSV UPLOAD & VISUALIZATION
# ---------------------------------------------------------
with tab1:
    st.header("Bulk Prediction")
    st.write("Upload a dataset to generate multiple predictions at once.")
    
    uploaded_file = st.file_uploader("Upload CSV with Housing Features", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Data type enforcement
            for col in ["MSSubClass", "MoSold"]:
                if col in df.columns:
                    df[col] = df[col].astype(str)
                    
            missing_cols = set(expected_cols) - set(df.columns)
            if missing_cols:
                st.error(f"Schema Error: Missing required columns: {missing_cols}")
            else:
                X_input = df[expected_cols]
                log_predictions = pipeline.predict(X_input)
                actual_predictions = np.expm1(log_predictions)
                
                # Insert predictions at the front of the dataframe for easy viewing
                df.insert(0, 'Predicted_SalePrice', actual_predictions)
                
                st.success("✅ Predictions generated successfully!")
                
                # UI ENHANCEMENT: Display the results directly in the app
                st.subheader("📊 Prediction Results")
                
                # If only one row was uploaded, show a large price metric
                if len(df) == 1:
                    price = df['Predicted_SalePrice'].iloc[0]
                    st.metric(label="Estimated Sale Price", value=f"${price:,.2f}")
                
                # Display an interactive table in the UI
                st.dataframe(df.head(50), use_container_width=True)
                
                # Keep the download button for the full file
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download Full Predictions as CSV",
                    data=csv,
                    file_name='predicted_house_prices.csv',
                    mime='text/csv',
                )
                
        except Exception as e:
            st.error(f"Runtime Error: {str(e)}")


# ---------------------------------------------------------
# TAB 2: MANUAL DATA ENTRY FORM
# ---------------------------------------------------------
with tab2:
    st.header("Predict a Single House")
    st.info("Enter the features for a single property to get an instant price estimate.")
    
    # Create a dynamic form based on the model's expected columns
    with st.form("single_prediction_form"):
        st.write("### Property Details")
        
        # Organize the 75 inputs into 4 neat columns so it doesn't require endless scrolling
        cols = st.columns(4)
        input_data = {}
        
        # Loop through every required feature and generate a text box
        for i, col_name in enumerate(expected_cols):
            with cols[i % 4]:
                # Default value is set to "0" or "None" to prevent empty string errors
                input_data[col_name] = st.text_input(label=col_name, value="0")
                
        # Form submission button
        submit_button = st.form_submit_button("Generate Prediction", type="primary")
        
    if submit_button:
        try:
            # Convert the user's manual inputs into a dataframe
            input_df = pd.DataFrame([input_data])
            
            # Data type enforcement
            for col in ["MSSubClass", "MoSold"]:
                if col in input_df.columns:
                    input_df[col] = input_df[col].astype(str)
                    
            # Generate the prediction
            log_pred = pipeline.predict(input_df[expected_cols])
            price_pred = np.expm1(log_pred)[0]
            
            st.success("✅ Prediction complete!")
            
            # UI ENHANCEMENT: Show the final price as a massive, highlighted metric
            st.metric(label="Estimated Sale Price", value=f"${price_pred:,.2f}")
            
        except Exception as e:
            st.error(f"Prediction Error: Please check your input values. {str(e)}")
