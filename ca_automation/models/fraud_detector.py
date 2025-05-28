# models/fraud_detector.py
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler


def detect_fraud(df):
   try:
      if len(df) < 10:
         return pd.DataFrame()

      scaler = MinMaxScaler()
      scaled_data = scaler.fit_transform(df[["Amount"]])

      X = []
      for i in range(len(scaled_data) - 3):
         X.append(scaled_data[i:i + 3])
      X = np.array(X)

      model = Sequential([
         LSTM(50, activation='relu', input_shape=(3, 1)),
         Dense(1)
      ])
      model.compile(optimizer='adam', loss='mse')
      model.fit(X, scaled_data[3:], epochs=50, verbose=0)

      predictions = model.predict(X, verbose=0)
      errors = np.abs(predictions - scaled_data[3:])
      threshold = np.mean(errors) + 2 * np.std(errors)
      anomalies = df.iloc[3:][errors.flatten() > threshold]
      return anomalies
   except Exception as e:
      print(f"Error in detect_fraud: {e}")
      return pd.DataFrame()