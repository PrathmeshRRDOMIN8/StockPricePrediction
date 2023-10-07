import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from yahoo_fin.stock_info import get_data
from keras.models import load_model
import streamlit as st
from keras.layers import Dense,Dropout, LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
#importing yahoo api for fetching data



st.title('Stock Trend Prediction')
user_input = st.text_input('Enter Stock Ticker','AAPL')
sdi = st.text_input('Start Year (20XX)','2018')
smi = st.text_input('Start Month (MM)','01')
syi = st.text_input('Start Date (DD)','01')
edi = st.text_input('End Year (20XX)','2023')
emi = st.text_input('End Month (MM)','10')
eyi = st.text_input('End Date (DD)','01')

start = sdi + '/' + smi + '/' + syi
end = edi + '/' + emi + '/' + eyi


data = get_data(user_input, start_date=start, end_date=end, index_as_date = True, interval="1d")


#Describing Data
st.subheader('Data from ' + start + ' to ' + end)
st.write(data.describe())


#Visualizations
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(data.close)
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig)


#Visualizations
st.subheader('Closing Price vs Time Chart with 100 days Moving Average')
ma100 = data.close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(data.close)
plt.plot(ma100,'r')
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig)

#Visualizations
st.subheader('Closing Price vs Time Chart with 200 days Moving Average')
ma200 = data.close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(data.close)
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig)


st.subheader('Closing Price vs Time Chart with 200 days Moving Average')

dtrain = pd.DataFrame(data['close'][0:int(len(data)*0.70)])
dtest = pd.DataFrame(data['close'][int(len(data)*0.70):int(len(data))])
scaler = MinMaxScaler(feature_range=(0,1))
data_train_array = scaler.fit_transform(dtrain)


#Loading Model
model = load_model("keras_model.h5")
past_100_days = dtrain.tail(100)
final_df = past_100_days.append(dtest, ignore_index = True)
input_data = scaler.fit_transform(final_df)
x_test = []
y_test = []
for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
    
x_test,y_test = np.array(x_test), np.array(y_test)

#Making predictions

y_predicted = model.predict(x_test)

scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor


st.subheader('Predictions vs Original')
figg = plt.figure(figsize = (12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(figg)











