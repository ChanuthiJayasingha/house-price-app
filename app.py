import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings('ignore')

# 1. UI Configuration & Sidebar (Must be the first Streamlit commands)
st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")

with st.sidebar:
    st.header("🏠 Ames Housing Predictor")
    st.markdown("An advanced machine learning pipeline using XGBoost.")
    st.markdown("---")
    st.markdown("### 📌 Instructions:")
    st.markdown("**Bulk Predict:** Upload a CSV matching the training schema. Great for analyzing multiple properties.")
    st.markdown("**Single House:** Enter details manually into the categorized form to test specific feature combinations.")
    st.markdown("---")
    st.caption("Powered by Streamlit & Scikit-Learn")

st.title("🏠 XGBoost House Price Predictor")
st.markdown("Predict the sale price of houses in Ames, Iowa based on 75 unique property features.")

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
            
            # Data type enforcement to prevent pipeline crashes
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
                
                # Insert predictions at the front
                df.insert(0, 'Predicted_SalePrice', actual_predictions)
                st.success("✅ Predictions generated successfully!")
                
                st.subheader("📊 Prediction Results")
                if len(df) == 1:
                    price = df['Predicted_SalePrice'].iloc[0]
                    st.metric(label="Estimated Sale Price", value=f"${price:,.2f}")
                
                st.dataframe(df.head(50), use_container_width=True)
                
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
# TAB 2: MANUAL DATA ENTRY FORM (GROUPED)
# ---------------------------------------------------------
with tab2:
    st.header("Predict a Single House")
    st.info("Expand the categories below to enter property details. Default values are set to 0.")
    
    with st.form("single_prediction_form"):
        # Create expandable containers for logical UI grouping
        exp_location = st.expander("📍 Location & Zoning", expanded=True)
        exp_size = st.expander("📏 Dimensions & Size")
        exp_quality = st.expander("✨ Quality & Condition")
        exp_amenities = st.expander("🏊 Amenities (Garage, Basement, Pool, etc.)")
        exp_other = st.expander("🔧 Other Features")
        
        input_data = {}
        
        # Dynamically distribute the 75 inputs into the correct expander
        for col in expected_cols:
            col_lower = col.lower()
            
            # Routing logic based on feature name keywords
            if any(k in col_lower for k in ['area', 'sf', 'room']):
                target_exp = exp_size
            elif any(k in col_lower for k in ['qual', 'cond']):
                target_exp = exp_quality
            elif any(k in col_lower for k in ['garage', 'bsmt', 'pool', 'fire', 'porch', 'deck']):
                target_exp = exp_amenities
            elif any(k in col_lower for k in ['mszoning', 'neigh', 'street', 'alley', 'lot']):
                target_exp = exp_location
            else:
                target_exp = exp_other
                
            # Render the input inside the targeted expander
            with target_exp:
                input_data[col] = st.text_input(label=col, value="0", key=col)
                
        submit_button = st.form_submit_button("Generate Prediction", type="primary")
        
    if submit_button:
        try:
            input_df = pd.DataFrame([input_data])
            
            # Data type enforcement
            for col in ["MSSubClass", "MoSold"]:
                if col in input_df.columns:
                    input_df[col] = input_df[col].astype(str)
                    
            log_pred = pipeline.predict(input_df[expected_cols])
            price_pred = np.expm1(log_pred)[0]
            
            st.success("✅ Prediction complete!")
            st.metric(label="Estimated Sale Price", value=f"${price_pred:,.2f}")
            
        except Exception as e:
            st.error(f"Prediction Error: Please check your input values. {str(e)}")
