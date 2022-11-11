import pandas as pd
import streamlit as st
import numpy as np
import pickle
from prophet import Prophet

st.title('Gold price forecasting')

st.write("Note: For getting result select number of forcasting days")

data=pd.read_csv('df.csv')

appdata=data
appdata['ds']= pd.to_datetime(appdata['ds'],errors='coerce')

max_date=appdata['ds'].max()

st.write("Select Forecast Period")
Periods_input=st.number_input('How Many Days Forecast you want?',
min_value=1,max_value=365)

if data is not None:
    obj=Prophet()
    obj.fit(appdata)


    st.write("Visualize forecasted data")
    st.write("The following plot shows future predicted values.'yhat is the predicted values ; upper and lower limits are 80% confidence intervals by default")
if data is not None:
        future=obj.make_future_dataframe(periods=Periods_input)

        fcst=obj.predict(future)
        forecast=fcst[['ds','yhat','yhat_lower','yhat_upper']]
        forecast_filtered=forecast[forecast['ds']>max_date]
        st.write(forecast_filtered)
        st.write("The next Visual Shows Actual & Predictedc values over time.")

        figure1=obj.plot(fcst)
        st.write(figure1)
        st.write("The following plots show a high level trend of predicted values, The day of week trends & yearly trends .Blue shades area represents upper & lower confidence intervals.")

        figure2=obj.plot_components(fcst)
        st.write(figure2)

