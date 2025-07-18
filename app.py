import joblib
import streamlit as st
import pandas as pd
from PIL import Image
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import threading

# Load Model & Scaler
model = joblib.load('model.pkl')
en = joblib.load('en.pkl')
df_main = pd.read_csv('data.csv', encoding='unicode_escape')
image = Image.open('img.png')

# Calculation Functions
def cal_SOi(so2):
    if so2 <= 40:
        return so2 * (50 / 40)
    elif so2 <= 80:
        return 50 + (so2 - 40) * (50 / 40)
    elif so2 <= 380:
        return 100 + (so2 - 80) * (100 / 300)
    elif so2 <= 800:
        return 200 + (so2 - 380) * (100 / 420)
    elif so2 <= 1600:
        return 300 + (so2 - 800) * (100 / 800)
    return 400 + (so2 - 1600) * (100 / 800)

def cal_Noi(no2):
    if no2 <= 40:
        return no2 * (50 / 40)
    elif no2 <= 80:
        return 50 + (no2 - 40) * (50 / 40)
    elif no2 <= 180:
        return 100 + (no2 - 80) * (100 / 100)
    elif no2 <= 280:
        return 200 + (no2 - 180) * (100 / 100)
    elif no2 <= 400:
        return 300 + (no2 - 280) * (100 / 120)
    return 400 + (no2 - 400) * (100 / 120)

def cal_SPMi(spm):
    if spm <= 50:
        return spm * (50 / 50)
    elif spm <= 100:
        return 50 + (spm - 50) * (50 / 50)
    elif spm <= 250:
        return 100 + (spm - 100) * (100 / 150)
    elif spm <= 350:
        return 200 + (spm - 250) * (100 / 100)
    elif spm <= 430:
        return 300 + (spm - 350) * (100 / 80)
    return 400 + (spm - 430) * (100 / 430)

def cal_aqi(si, ni, spmi):
    return max(si, ni, spmi)

def predict_logic(state, so2, no2, spm):
    state_encoded = en.transform([state])[0]
    soi = cal_SOi(so2)
    noi = cal_Noi(no2)
    spmi = cal_SPMi(spm)
    aqi = cal_aqi(soi, noi, spmi)
    test = [state_encoded, soi, noi, spmi]
    prediction = model.predict([test])[0]
    return aqi, prediction

# Streamlit UI
def streamlit_app():
    st.image(image, width=650)
    st.title('Air Pollution Prediction')
    st.markdown("<h2 style='color: red; text-align: center;'>Please Enter Input</h2>", unsafe_allow_html=True)
    state = st.selectbox("Select Your State", df_main['state'].unique())
    so2 = st.number_input('Enter Your Area SO2', value=0.0)
    no2 = st.number_input('Enter Your Area NO2', value=0.0)
    spm = st.number_input('Enter Your Area PM10', value=0.0)

    if st.button('Predict'):
        aqi, prediction = predict_logic(state, so2, no2, spm)
        st.write(f"AQI in your area is {aqi:.2f}")
        st.write(f"Prediction: {prediction}")

# FastAPI Backend
api_app = FastAPI()

class AQIRequest(BaseModel):
    state: str
    so2: float
    no2: float
    spm: float

@api_app.post("/predict_aqi")
def predict_aqi(request: AQIRequest):
    if request.state not in df_main['state'].unique():
        return {"error": "Invalid state name."}
    aqi, prediction = predict_logic(request.state, request.so2, request.no2, request.spm)
    return JSONResponse(content={
        "state": request.state,
        "AQI Value": round(aqi, 2),
        "Prediction": prediction
    })

# Run Streamlit and FastAPI concurrently using threading
def run_fastapi():
    import uvicorn
    uvicorn.run(api_app, host="0.0.0.0", port=8000)

if __name__ == '__main__':
    thread = threading.Thread(target=run_fastapi, daemon=True)
    thread.start()
    streamlit_app()


