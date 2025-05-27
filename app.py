import streamlit as st
import pickle
import numpy as np
import sqlite3
from datetime import datetime

# Load the trained model
with open("model.pkl", "rb") as f:
    model, feature_names = pickle.load(f)

# Database setup
def create_table():
    conn = sqlite3.connect("patient_data.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS patients
                 (name TEXT, age INTEGER, gender TEXT, symptoms TEXT,
                 prediction TEXT, date TEXT)''')
    conn.commit()
    conn.close()

def insert_data(name, age, gender, symptoms, prediction):
    conn = sqlite3.connect("patient_data.db")
    c = conn.cursor()
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO patients VALUES (?, ?, ?, ?, ?, ?)",
              (name, age, gender, symptoms, prediction, date))
    conn.commit()
    conn.close()

create_table()

# Streamlit UI
st.set_page_config(page_title="Medical Diagnosis System", layout="centered")
st.title("🩺 Medical Assistant & Diagnosis System")

# User details
st.subheader("🧍‍ Patient Info")
name = st.text_input("Full Name")
age = st.number_input("Age", min_value=0, max_value=120, step=1)
gender = st.radio("Gender", ["Male", "Female", "Other"])

# Symptom input
st.subheader("🤒 Select Your Symptoms")
selected_symptoms = st.multiselect("Symptoms:", feature_names)

# Image upload (optional)
st.subheader("🖼️ Upload an image (e.g., skin condition)")
uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])
if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.info("🧠 Note: Image processing is not yet active in this version.")

# Prediction
if st.button("🔍 Diagnose"):
    input_vector = [1 if symptom in selected_symptoms else 0 for symptom in feature_names]
    prediction = model.predict([input_vector])[0]

    # Save to database
    insert_data(name, age, gender, ", ".join(selected_symptoms), prediction)

    st.success(f"✅ Predicted Disease: **{prediction}**")
    st.info("💡 For serious symptoms, please consult a doctor.")
