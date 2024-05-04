import ta
import yfinance as yf
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
import requests
from datetime import date, timedelta

class Prediction:
    
    @staticmethod
    def send_message(bot_token, bot_chatID, message):
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        data = {"chat_id": bot_chatID, "text": message, "parse_mode": "Markdown"}
        response = requests.post(url, data=data)
        return response.json()

    @staticmethod
    def predict_next_close(sym):
        try:
            data = yf.download(sym, period='3y')
            
            data['Close_next'] = data['Close'].shift(-1)
            data['Close_pct_change'] = data['Close_next'].pct_change() *100
            data = data.fillna(0)

            # Calculate technical indicators
            data = pd.DataFrame(ta.add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True))
            index = pd.DataFrame(data.columns)
            index = index[0].to_numpy()[8:]
            index = index.tolist()
            X = data[index].values
            y = data[['Close_next', 'Close_pct_change']].values

            # Train the KNN model
            model = KNeighborsRegressor()
            model.fit(X, y)

            # Prepare feature data for the next day
            last_day_data = data.iloc[-1][index].values.reshape(1, -1)  # Use the last day's technical indicator values

            # Predict the next closing price and percentage change
            next_close, next_pct_change = model.predict(last_day_data)[0]

            return next_close, next_pct_change

        except Exception as e:
            print(f"Error occurred: {e}")
            return None


if __name__ == '__main__':
    #aksd
    bot_token = '6111156932:AAGWg7uRvD7cwTK_UpzGcWv54MJkKWgyMoI'
    bot_chatID = '5562607566'
    today = date.today()
    two_days_later = today + timedelta(days=1)
    message = f"predictions for {two_days_later} using Day timeframe"
    Prediction.send_message(bot_token, bot_chatID, message)
    ok = pd.read_csv("100_tick.csv")
    symbols = ok['Symbol']
    for sym in symbols:
        prediction = Prediction.predict_next_close(sym)
        if prediction and abs(prediction[1]) > 1.0:
            message = f"{sym}\nchange: {prediction[1]:.2f}%"
            Prediction.send_message(bot_token, bot_chatID, message)
