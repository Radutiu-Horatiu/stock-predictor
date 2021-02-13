# Yfinance test

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

stock_name = "TSLA"
period = "1y"

stock = yf.Ticker(stock_name)

# Read the dataset:
df = stock.history(period=period)
final_dataset = df.iloc[:, 3].values

new_dataset = pd.DataFrame(index=df.index,columns=['Close'])
for i in range(0,len(df)):
    new_dataset["Close"][i] = df["Close"][i]

# Splitting in train and test data
train_data = final_dataset[:round(0.8 * len(final_dataset))]
test_data = final_dataset[round(0.8 * len(final_dataset)):]

# Normalize the new filtered dataset:
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(final_dataset.reshape(-1, 1))

# Scaling dates
x_train_data,y_train_data=[],[]
for i in range(round(len(train_data)/2),len(train_data)):
    x_train_data.append(scaled_data[i-round(len(train_data)/2):i,0])
    y_train_data.append(scaled_data[i,0])
    
x_train_data,y_train_data = np.array(x_train_data), np.array(y_train_data)
x_train_data = np.reshape(x_train_data,(x_train_data.shape[0], x_train_data.shape[1],1))

#  Build and train the LSTM model:
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense

lstm_model=Sequential()
lstm_model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train_data.shape[1],1)))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(1))

lstm_model.compile(loss='mean_squared_error', optimizer='adam')
lstm_model.fit(x_train_data,y_train_data,epochs=10,batch_size=1,verbose=2)

inputs_data = new_dataset[len(new_dataset)-len(test_data)-round(len(train_data)/2):].values
inputs_data = inputs_data.reshape(-1,1)
inputs_data = scaler.transform(inputs_data)

# Take a sample of a dataset to make stock price predictions using the LSTM model:
X_test=[]
for i in range(round(len(train_data)/2),inputs_data.shape[0]):
    X_test.append(inputs_data[i-round(len(train_data)/2):i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
predicted_closing_price = lstm_model.predict(X_test)
predicted_closing_price = scaler.inverse_transform(predicted_closing_price)

# Visualize the predicted stock costs with actual stock costs:
train_data = new_dataset[:round(0.8 * len(new_dataset))]
test_data = new_dataset[round(0.8 * len(new_dataset)):]
test_data['Predictions'] = predicted_closing_price
plt.plot(train_data["Close"]) # actual data
plt.plot(test_data[['Close',"Predictions"]]) # predicted
plt.title(stock.info['longName'])
plt.show()