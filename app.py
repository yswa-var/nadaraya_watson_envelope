import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st



        
class Backtest:
    def __init__(self, symbol='ACC.NS', tim='1y', sd=9):
        self.tim = tim
        self.symbol = symbol
        self.df = yf.download(tickers=symbol, period = self.tim)
        self.src = self.df["Close"].values
        self.h = sd
        y2, y1 = self.nadaraya_watson_envelope()
        self.gen_signals(y1,y2)


    def nadaraya_watson_envelope(self):
        n = len(self.src)
        y2 = np.empty(n)
        y1 = np.empty(n)
        h= self.h
        for i in range(n):
            sum = 0
            sumw = 0
            for j in range(n):
                w = np.exp(-(np.power(i-j,2)/(h*h*2)))
                sum += self.src[j]*w
                sumw += w
            y2[i] = sum/sumw
            if i > 0:
                y1[i] = (y2[i] + y2[i-1]) / 2
        self.df['y2'] = y2
        self.df['y1'] = y1
        return y2, y1
    
    def gen_signals(self,y1,y2):
        buy_signals = []
        sell_signals = []
        thld = 0.01

        for i in range(1, len(y2)):
            d = y2[i] - y2[i-1]
            if d > thld and y2[i-1] < y1[i-1]:
                buy_signals.append(i)
            elif d < -thld and y2[i-1] > y1[i-1]:
                sell_signals.append(i)
        money = 100
        trades = 0
        profit = []
        for i in range(len(buy_signals)):
            buy_index = buy_signals[i]
            if i < len(sell_signals):
                sell_index = sell_signals[i]
                trades += 1
                money *= self.src[sell_index] / self.src[buy_index]
                profit.append(money - 100)
        self.profit  = pd.DataFrame(profit)
        self.rets = "Returns "+ self.tim +" = " + str(round(((money/100-1)*100),2)) + "%"
        self.trades = "Total Trades: " + str(trades)
        self.roi = "Total Return: " + str(round((money-100),2)) + "%"
        self.avg_return = "Average Return Per Trade: " + str(round((money-100)/trades,2)) + "%"
        self.win_rate = "Win Rate: " + str(round((len([x for x in profit if x > 0])/trades)*100,2)) + "%"
        plt.figure(figsize=(30,10))
        plt.axis('off')
        plt.plot(y2,color='blue')
        plt.plot(self.src,color='black', label='close')
        for signal in buy_signals:
            plt.axvline(x=signal, color='green',linewidth=2)

        for signal in sell_signals:
            plt.axvline(x=signal, color='red',linewidth=2)
        plt.legend()
        plt.show()
ticks = pd.read_csv("100_tick.csv")
clck = ticks['Symbol'].values
selected_option = st.selectbox("Select an Stock",clck)
st.write("You selected: ", selected_option)
tim = "1y" #st.text_input("time frame ex: 1y,2y,100d,10d..")
std = 8 #st.text_input("Standard devation, like 5,6,7,8,9....")
backtest = Backtest(selected_option, tim, int(std))
st.pyplot(plt)
profits = backtest.profit.iloc[:, 0]
pro = backtest.profit # Remove extra column

metrics = {
    "Total Trades": len(profits),
    "Total Return": round(profits.iloc[-1], 2),
    "Average Daily Return": round(profits.mean(), 2),
    "Std Dev of Daily Returns": round(pro.iloc[:, 0].std(), 2),
    "Sharpe Ratio": round((backtest.profit.iloc[:, 0].mean()) / backtest.profit.iloc[:, 0].std(), 2),
    "Max Drawdown": round(profits.min(), 2),
    "Winning Trades": round(len(profits[profits > 0]), 2),
    "Losing Trades": round(len(profits[profits < 0]), 2),
    "Profit Factor": round(abs(profits[profits > 0].sum() / 2), 2)
}
df = pd.DataFrame.from_dict(metrics,orient="index", columns=["Value"])
print(df)
st.sidebar.write("Backtest Metrics")
st.sidebar.table(df)
st.write(" [How we made this](https://github.com/bbmusa/nadaraya_watson_envelope)")