# Stock Price prediction software and application using python


# Importing the modules

import numpy as np;
import matplotlib.pyplot as plt
import pandas_datareader as data
import pandas as pd 
from keras.models import load_model
import streamlit as st

# selecting the time-frame 


start = '2014-01-01'
end = '2021-12-10'

# importing the data from yahoofinance

st.title('Stock Price Prediction & Trend Analysis')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = data.DataReader(user_input, 'yahoo', start, end)

# describing data
st.subheader('Data from 2010 - 2019')
st.write(df.describe())

# Visualization

st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100 & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)

# splitting data into training(70%) and testing(30%)

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

# For the LSTM Model we have to scale the data

# scaling in 0 and 1

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 1))

# convert training data into an array

data_training_array = scaler.fit_transform(data_training)

# define the data into x train and y train

x_train = []
y_train = []

# price of a stock cannot move very largely it will be in the range of previous few days

# creating x_train to store previous 100 days data dynamically

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i, 0])
    
# convert x_train and y_train to numpy arrays
# so that we can provide this data to LSTM


x_train, y_train = np.array(x_train), np.array(y_train)

# load my model

model = load_model('keras_model.h5')

# Testing part

past_100_days = data_training.tail(100)

final_df = past_100_days.append(data_testing, ignore_index = True)

# Scaling final_df data

input_data = scaler.fit_transform(final_df)

# Defining testing data sets

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Making Predictions

y_predicted = model.predict(x_test)

#scaler
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

#Plotting the predicted and original value
# Final graph

st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize = (12, 6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)










