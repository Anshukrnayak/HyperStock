import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler


def forecast_expenses(df):
   try:
      expenses = df[df["Category"] == "expense"][["Date", "Amount"]].groupby("Date").sum().reset_index()
      if len(expenses) < 10:
         return pd.DataFrame({"Date": [], "Amount": []})

      scaler = MinMaxScaler()
      scaled_data = scaler.fit_transform(expenses[["Amount"]])

      X, y = [], []
      for i in range(len(scaled_data) - 3):
         X.append(scaled_data[i:i + 3])
         y.append(scaled_data[i + 3])
      X, y = np.array(X), np.array(y)

      model = Sequential([
         LSTM(50, activation='relu', input_shape=(3, 1)),
         Dense(1)
      ])
      model.compile(optimizer='adam', loss='mse')
      model.fit(X, y, epochs=50, verbose=0)

      last_sequence = scaled_data[-3:]
      forecast = model.predict(np.array([last_sequence]), verbose=0)
      forecast = scaler.inverse_transform(forecast)[0][0]

      last_date = pd.to_datetime(expenses["Date"].iloc[-1])
      forecast_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 4)]
      forecast_values = [forecast] * 3
      return pd.DataFrame({"Date": forecast_dates, "Amount": forecast_values})
   except Exception as e:
      print(f"Error in forecast_expenses: {e}")
      return pd.DataFrame({"Date": [], "Amount": []})
