import streamlit as st
import pandas as pd
import joblib
import numpy as np

model = joblib.load("xgboost_ev_model.pkl")

st.title("EV Registration Forecast Using XGBoost")

year_input = st.number_input("Enter Model Year:", min_value=2010, max_value=2030, value=2025)
pred = model.predict(np.array([[year_input]]))
st.write(f"Predicted EV Count for {year_input}: {int(pred[0])}")