# Discription : This Program Uses An Artificial Neural Network Called Long Short Term Memory (LSTM)
# To Predict The Closing Stock Price Of A Corporation (Apple Inc.) Using Past 60 Days Stock Price.

import math
from tracemalloc import start
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# Get The Stock Quote
df = web.DataReader('AAPL', data_source='yahoo',
                    start='2012-01-01', end='2019-12-17')
# Show The Data
#print(df)

# Get The Number Of Rows And Colums In The Data Set
# print(df.shape)

#Visualize The Closing Price History
plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price USD ($)',fontsize=18)
#plt.show()

#Create a new DataFrame With Only The "clsoe Column"
data = df.filter(['Close'])

#Convert The DtatFrame To A Numpy Array
dataset = data.values

#Get The Number Of Rows To Train The Model On
training_data_len = math.ceil(len(dataset) *.8)
#print(training_data_len)

#Scale The Data
scaler =MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
print(scaled_data)

