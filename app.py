import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import warnings

warnings.filterwarnings('ignore')

# 1. Page Configuration
st.set_page_config(page_title="Ames Housing Predictor", layout="wide")
st.title("🏡 House Price Predictor")
st.write("Estimate Ames Housing prices using either bulk CSV upload or the interactive manual entry tool.")

# 2. Load New Artifacts
@st.cache_resource
def load_models():
    # Updated to load the new "top feature" models and JSON schema
    pipeline = joblib.load('xgb_top_housing_pipeline.joblib')
    expected_cols = joblib.load('top_feature_columns.joblib')
    with open('feature_schema.json', 'r') as f:
        schema = json.load(f)
    return pipeline, expected_cols, schema

try:
    pipeline, expected_cols, feature_schema = load_models()
except Exception as e:
    st.error(f"Error loading models: {str(e)}. Ensure 'xgb_top_housing_pipeline.joblib', 'top_feature_columns.joblib', and 'feature_schema.json' are uploaded.")
    st.stop()

# 3. Setup Tabs
tab1, tab2 = st.tabs(["Bulk Predict (CSV)", "Single House (Manual)"])

# ==========================================
# TAB 1: BULK PREDICTION (Your Original Logic, Updated for Tabs)
# ==========================================
with tab1:
    st.header("Bulk Prediction")
    uploaded_file = st.file_uploader("Upload CSV with Housing Features", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Enforce exact data types from training (preserved from your original code)
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

# ==========================================
# TAB 2: SINGLE HOUSE SMART WIDGETS
# ==========================================
with tab2:
    st.header("Predict Single Property")
    st.markdown("Adjust the key features below to estimate the property value.")
    
    user_input = {}
    
    # Map technical column names to human-readable UI labels
    friendly_names = {
        "GrLivArea": "Above Ground Living Area (sq ft)",
        "TotalBsmtSF": "Total Basement Area (sq ft)",
        "1stFlrSF": "First Floor Area (sq ft)",
        "GarageArea": "Garage Area (sq ft)",
        "LotArea": "Lot Size (sq ft)",
        "OverallQual": "Overall Quality (1-10)",
        "OverallCond": "Overall Condition (1-10)",
        "YearBuilt": "Original Construction Year",
        "YearRemodAdd": "Remodel / Addition Year",
        "BsmtFinSF1": "Finished Basement Area (sq ft)",
        "WoodDeckSF": "Wood Deck Area (sq ft)",
        "OpenPorchSF": "Open Porch Area (sq ft)",
        "Fireplaces": "Number of Fireplaces",
        "Neighborhood": "Neighborhood / Location",
        "BldgType": "Building Type",
        "MSSubClass": "Building Class",
        "MoSold": "Month Sold"
    }
    
    categories = {
        "Dimensions & Size": ["GrLivArea", "TotalBsmtSF", "1stFlrSF", "GarageArea", "LotArea"],
        "Quality & Condition": ["OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd"],
        "Amenities & Other": ["BsmtFinSF1", "WoodDeckSF", "OpenPorchSF", "Fireplaces", "Neighborhood"]
    }
    
    # Dynamically generate grouped smart widgets based on JSON schema
    for category_name, expected_category_cols in categories.items():
        with st.expander(f"📌 {category_name}", expanded=True):
            cols = st.columns(3)
            col_idx = 0
            
            for feature in expected_category_cols:
                if feature in feature_schema and feature in expected_cols:
                    schema_data = feature_schema[feature]
                    display_label = friendly_names.get(feature, feature) 
                    
                    with cols[col_idx % 3]:
                        if schema_data["type"] == "categorical":
                            user_input[feature] = st.selectbox(
                                label=display_label,
                                options=schema_data["options"],
                                index=schema_data["options"].index(schema_data["default"]) if schema_data["default"] in schema_data["options"] else 0
                            )
                        elif schema_data["type"] == "numerical":
                            user_input[feature] = st.number_input(
                                label=display_label,
                                min_value=float(schema_data["min"]),
                                max_value=float(schema_data["max"] * 1.5), 
                                value=float(schema_data["default"]),
                                step=1.0
                            )
                    col_idx += 1
    
    # Catch any remaining top features not explicitly categorized above
    uncategorized = [f for f in expected_cols if f not in [item for sublist in categories.values() for item in sublist]]
    if uncategorized:
        with st.expander("📌 Additional Features", expanded=True):
            cols = st.columns(3)
            for idx, feature in enumerate(uncategorized):
                schema_data = feature_schema[feature]
                display_label = friendly_names.get(feature, feature)
                
                with cols[idx % 3]:
                    if schema_data["type"] == "categorical":
                        user_input[feature] = st.selectbox(display_label, schema_data["options"])
                    else:
                        user_input[feature] = st.number_input(display_label, value=float(schema_data["default"]))

    # Generate Prediction Button
    if st.button("Predict Price", type="primary"):
        input_df = pd.DataFrame([user_input])
        
        # Enforce specific data types for manual input if they were caught in the schema
        for col in ["MSSubClass", "MoSold"]:
            if col in input_df.columns:
                input_df[col] = input_df[col].astype(str)
                
        # Ensure columns match training order exactly
        input_df = input_df[expected_cols]
        
        try:
            log_pred = pipeline.predict(input_df)[0]
            predicted_price = np.expm1(log_pred)
            st.success(f"### Estimated Sale Price: ${predicted_price:,.2f}")
        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")
