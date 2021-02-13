import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf

# Preprocessing
stock = yf.Ticker("VOO")
df = stock.history(period="max")
df = df[["Close"]]

# Scaling
from sklearn.preprocessing import MinMaxScaler
train = df
scaler = MinMaxScaler()
scaler.fit(train)
train = scaler.transform(train)

from keras.preprocessing.sequence import TimeseriesGenerator
n_input = 12
n_features = 1
generator = TimeseriesGenerator(train, train, length=n_input, batch_size=6)

#  Build and train the LSTM model
import keras
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense
model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(n_input, n_features)))
model.add(Dropout(0.15))
model.add(Dense(1))
optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse')
history = model.fit_generator(generator,epochs=50,verbose=1)

# Predicting on future dates
pred_list = []
batch = train[-n_input:].reshape((1, n_input, n_features))
for i in range(n_input):   
    pred_list.append(model.predict(batch)[0]) 
    batch = np.append(batch[:,1:,:],[[pred_list[i]]],axis=1)
    
add_dates = [df.index[-1] + pd.DateOffset(months=x) for x in range(0,13) ]
future_dates = pd.DataFrame(index=add_dates[1:],columns=df.columns)

df_predict = pd.DataFrame(scaler.inverse_transform(pred_list),
                          index=future_dates[-n_input:].index, columns=['Prediction'])
df_proj = pd.concat([df,df_predict], axis=1)

# Visualization
plt.plot(df_proj.index, df_proj['Close'])
plt.plot(df_proj.index, df_proj['Prediction'])
plt.title(stock.info['longName'])
plt.show()