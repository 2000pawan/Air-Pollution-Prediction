# Import Important Library.

import joblib
import streamlit as st 
from PIL import Image
import pandas as pd


# Load Model & Scaler

model=joblib.load('model.pkl')
en=joblib.load('en.pkl')

# load datase

df_main=pd.read_csv('data.csv',encoding='unicode_escape')

# Load Image

image=Image.open('img.png')

# Streamlit Function For Building Button & app.

def main():
    st.image(image,width=650)
    st.title('Air Pollution Prediction')
    html_temp='''
    <div style='background-color:red; padding:12px'>
    <h1 style='color:  #000000; text-align: center;'>Air Pollution Prediction Machine Learning Model</h1>
    <h3 style='color:  #000000; text-align: center;'>This Model Consider Only Three Main Feature Sulfur Dioxide,Nitrogen Dioxide,PM10(Particulate Matter) If Your area AQI Different From this.That means it depend on different parameter.</h3>
    <h2 style='color:  #000000; text-align: right;'>PAWAN YADAV</h2>
    <h4 style='color:  #000000; text-align: right;'>Machine Learning Engineer</h4>
    </div>
    <h2 style='color:  red; text-align: center;'>Please Enter Input</h2>
    '''
    st.markdown(html_temp,unsafe_allow_html=True)
    state= st.selectbox("Select Your State",df_main['state'].unique())
    so2=st.number_input('Enter Your Area SO2(Sulfur Dioxide).',value=None)
    no2=st.number_input('Enter Your Area NO2(Nitrogen Dioxide)).',value=None)
    spm=st.number_input('Enter Your Area Particulate Matter(PM10 (it means inhalable pollutant particles with a diameter less than 10 micrometers).',value=None)
    input=[state,so2,no2,spm]
    result=''
    if st.button('Predict',''):
        result=prediction(input)
    temp='''
     <div style='background-color:navy; padding:8px'>
     <h1 style='color: gold  ; text-align: center;'>{}</h1>
     </div>
     '''.format(result)
    st.markdown(temp,unsafe_allow_html=True)
    


# Prediction Function to predict from model.

def cal_SOi(so2):  
    si=0  
    if (so2<=40):  
     si= so2*(50/40)  
    elif (so2>40 and so2<=80):  
     si= 50+(so2-40)*(50/40)  
    elif (so2>80 and so2<=380):  
     si= 100+(so2-80)*(100/300)  
    elif (so2>380 and so2<=800):  
     si= 200+(so2-380)*(100/420)  
    elif (so2>800 and so2<=1600):  
     si= 300+(so2-800)*(100/800)  
    elif (so2>1600):  
     si= 400+(so2-1600)*(100/800)
    return si 
def cal_Noi(no2):  
    ni=0  
    if(no2<=40):  
     ni= no2*50/40  
    elif(no2>40 and no2<=80):  
     ni= 50+(no2-40)*(50/40)  
    elif(no2>80 and no2<=180):  
     ni= 100+(no2-80)*(100/100)  
    elif(no2>180 and no2<=280):  
     ni= 200+(no2-180)*(100/100)  
    elif(no2>280 and no2<=400):  
     ni= 300+(no2-280)*(100/120)  
    else:  
     ni= 400+(no2-400)*(100/120)  
    return ni   
def cal_SPMi(spm):  
    spi=0  
    if(spm<=50):  
     spi=spm*50/50  
    elif(spm>50 and spm<=100):  
     spi=50+(spm-50)*(50/50)  
    elif(spm>100 and spm<=250):  
     spi= 100+(spm-100)*(100/150)  
    elif(spm>250 and spm<=350):  
     spi=200+(spm-250)*(100/100)  
    elif(spm>350 and spm<=430):  
     spi=300+(spm-350)*(100/80)  
    else:  
     spi=400+(spm-430)*(100/430)  
    return spi 
def cal_aqi(si,ni,spmi):  
    aqi=0  
    if(si>ni  and si>spmi):  
     aqi=si  
    if(ni>si and ni>spmi):  
     aqi=ni   
    if(spmi>si and spmi>ni):  
     aqi=spmi  
    return aqi  

def prediction(input):
    state=(en.transform([input[0]]))[0]
    soi=cal_SOi(input[1])
    noi=cal_Noi(input[2])
    spm=cal_SPMi(input[3])
    test=[state,soi,noi,spm]
    aqi=cal_aqi(soi,noi,spm)
    predict=model.predict([test])
    if predict[0]=='Good':
        return (f'AQI in your area is {aqi}',
            "AQI is Good It's great news that the air quality is good! Enjoy outdoor activities and take advantage of the fresh air. However, it's still good practice to monitor updates periodically.")
    elif predict[0]=='Poor':
        return (f'AQI in your area is {aqi}',
            "AQI is Poor individuals with respiratory conditions or sensitive health should limit outdoor exposure, especially during peak pollution times. Consider using air purifiers indoors.")
    elif predict[0]=='Moderate':
        return(f'AQI in your area is {aqi}',
            "AQI is Moderate while the air quality is acceptable, sensitive individuals may experience slight irritation. Consider reducing prolonged or heavy exertion outdoors.")
    elif predict[0]=='Unhealthy':
        return(f'AQI in your area is {aqi}',
            "AQI is Unhealthy individuals may experience health effects, especially those with existing conditions. Minimize outdoor activities, and if possible, stay indoors with filtered air.")
    elif predict[0]=='Very unhealthy.':
        return (f'AQI in your area is {aqi}',
            "AQI is Very unhealthy everyone may experience adverse health effects. Avoid outdoor activities, and if indoors, use air purifiers to improve air quality.")
    else:
        return (f'AQI in your area is {aqi}',
            "AQI is Hazardous it's highly dangerous to be outdoors. Stay indoors, and if necessary to go outside, wear masks and minimize exposure time.")


if __name__=='__main__':
    main()


