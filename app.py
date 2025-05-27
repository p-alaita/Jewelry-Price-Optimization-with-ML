import streamlit as st
import joblib
import pandas as pd
import os

# --- Model explanations for the dashboard ---
model_descriptions = {
    "LinearRegression": "A fast, simple, and interpretable model. Useful if you want to understand how each feature affects price.",
    "RandomForest": "A robust ensemble method. Works well with non-linear data and can handle feature interactions.",
    "XGBoost": "A high-performance boosting model. Great for maximizing prediction accuracy, even with complex relationships."
}

st.title("Jewelry Price Prediction App")

# --- Model selector ---
model_choice = st.sidebar.selectbox(
    "Select Model", 
    ["LinearRegression", "RandomForest", "XGBoost"]
)

st.write("### Model Description")
st.info(model_descriptions[model_choice])

# --- User input section (customize these fields based on your actual features) ---
st.write("#### Input Jewelry Features:")

main_gem = st.selectbox("Main Gem", ["Diamond", "Ruby", "Emerald", "Sapphire"])  # Update with your unique values!
main_metal = st.selectbox("Main Metal", ["Gold", "Silver", "Platinum"])
product_gender = st.selectbox("Product Gender", ["Women", "Men", "Unisex"])
# Add more features as needed!
weight = st.number_input("Weight (carats)", min_value=0.0, step=0.01)

user_input = {
    "Main gem": [main_gem],
    "Main metal": [main_metal],
    "Product gender": [product_gender],
    "Weight": [weight],
    # Add other required fields here!
}
input_df = pd.DataFrame(user_input)

# --- Prediction logic with error handling ---
if st.button("Predict Price"):
    try:
        pipeline_path = f"{model_choice}_pipeline.pkl"
        if not os.path.isfile(pipeline_path):
            st.error(f"Model pipeline file `{pipeline_path}` not found. Please train and save this model first.")
        else:
            pipeline = joblib.load(pipeline_path)
            # Ensure columns match expected by pipeline
            expected_features = pipeline.named_steps['preprocessor'].transformers_[1][2] + pipeline.named_steps['preprocessor'].transformers_[0][2]
            missing_cols = set(expected_features) - set(input_df.columns)
            if missing_cols:
                st.error(f"Missing columns in your input: {missing_cols}")
            else:
                prediction = pipeline.predict(input_df)[0]
                st.success(f"Predicted Price: ${prediction:,.2f}")
                st.caption("This estimate is based on the selected model and your provided features.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
import streamlit as st
import joblib
import numpy as np

# Load the preprocessor and models
@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load("preprocessor.pkl")
    models = {
        "Linear Regression": joblib.load("Linear_Regression_pipeline.pkl"),
        "Random Forest": joblib.load("Random_Forest_pipeline.pkl"),
        "XGBoost": joblib.load("xgboost_pipeline.pkl")
    }
    return preprocessor, models

preprocessor, models = load_artifacts()

st.title("Jewelry Price Prediction App")
st.sidebar.header("Select Model")
model_name = st.sidebar.radio("Choose Model", list(models.keys()))

st.sidebar.markdown("---")
st.markdown("### Enter Features to Predict Price")

# EXAMPLE: Adapt these input fields to match your actual feature set!
brand_id = st.selectbox("Brand ID", [1, 2, 3, 4, 5])
product_gender = st.selectbox("Product Gender", ["Men", "Women", "Unisex"])
main_color = st.selectbox("Main Color", ["Gold", "Silver", "Rose Gold"])
main_metal = st.selectbox("Main Metal", ["Gold", "Silver", "Platinum"])
main_gem = st.selectbox("Main Gem", ["Diamond", "Ruby", "Sapphire", "None"])
order_month = st.slider("Order Month", 1, 12, 1)
order_weekday = st.slider("Order Weekday (0=Mon)", 0, 6, 0)
order_year = st.slider("Order Year", 2018, 2025, 2023)
is_weekend = st.selectbox("Is Weekend", [0, 1])
days_since_first_order = st.number_input("Days Since First Order", 0)
is_repeat_customer = st.selectbox("Is Repeat Customer", [0, 1])
is_holiday = st.selectbox("Is Holiday", [0, 1])
# ... Add more fields as per your feature set ...

if st.button("Predict Price"):
    # Build a single-row DataFrame for your features
    import pandas as pd
    X = pd.DataFrame([{
        'Brand ID': brand_id,
        'Product gender': product_gender,
        'Main Color': main_color,
        'Main metal': main_metal,
        'Main gem': main_gem,
        'order_month': order_month,
        'order_weekday': order_weekday,
        'order_year': order_year,
        'is_weekend': is_weekend,
        'days_since_first_order': days_since_first_order,
        'is_repeat_customer': is_repeat_customer,
        'is_holiday': is_holiday,
        # Add other fields as needed...
    }])

    # PREPROCESS: transform features
    try:
        X_processed = preprocessor.transform(X)
    except Exception as e:
        st.error(f"Preprocessing error: {e}")
        st.stop()

    model = models[model_name]
    pred = model.predict(X_processed)
    st.success(f"Predicted Price: ${pred[0]:,.2f}")
