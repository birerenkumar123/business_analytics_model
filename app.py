import numpy as np
import pandas as pd
import pickle
import streamlit as st

# Load the trained model
@st.cache_resource
def load_model():
  with open("business_analytics_model.pkl", "rb") as file:
    model = pickle.load(file)
  return model


model = load_model()

st.title("Business Analytics: Customer Loyalty Prediction")
st.write(
    "Enter the transaction and customer details below to predict the customer"
    " loyalty score."
)

# Input fields based on your dataset columns
# Columns: ['Store_Location', 'Product_Category', 'Units_Sold', 'Revenue (£)', 'Discount_Applied (%)', 'Customer_Age']

store_options = {"London": 0, "Manchester": 1, "Leeds": 2, "Birmingham": 3}
category_options = {"Clothing": 0, "Electronics": 1, "Groceries": 2}

store_location_name = st.selectbox(
    "Store Location", options=list(store_options.keys())
)
product_category_name = st.selectbox(
    "Product Category", options=list(category_options.keys())
)

units_sold = st.number_input("Units Sold", min_value=1.0, max_value=50.0, value=10.0)
revenue = st.number_input(
    "Revenue (£)", min_value=0.0, max_value=1000.0, value=250.0
)
discount_applied = st.number_input(
    "Discount Applied (%)", min_value=0.0, max_value=100.0, value=15.0
)
customer_age = st.number_input(
    "Customer Age", min_value=18.0, max_value=100.0, value=35.0
)

# Map categorical values to numeric encoding used in training
store_location = store_options[store_location_name]
product_category = category_options[product_category_name]

# Prediction button
if st.button("Predict Loyalty Score"):
  # Create input array matching the training feature order
  # x = data.drop(columns='Loyalty_Score')
  # Features: Store_Location, Product_Category, Units_Sold, Revenue (£), Discount_Applied (%), Customer_Age
  input_data = np.array(
      [[
          store_location,
          product_category,
          units_sold,
          revenue,
          discount_applied,
          customer_age,
      ]]
  )

  # Note: During training, StandardScaler was applied to xtrain via sc.fit_transform(xtrain).
  # If your pipeline expects scaled inputs, ensure scaling matches the notebook.
  # Assuming the model expects standard scaled inputs or raw inputs depending on how pipeline was structured.
  # Since your GridSearchCV was fit on x1 (scaled features), you should scale input data if necessary,
  # or pass it directly if standard scaling parameters were baked into a pipeline.
  # (If you used standard scaler separately outside a Pipeline, you can apply the same transform here).

  prediction = model.predict(input_data)

  st.success(f"Predicted Loyalty Score (Scaled): {prediction[0]:.4f}")