import numpy as np
from sklearn.kernel_ridge import KernelRidge
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
        profit = []
        for i in range(len(buy_signals)):
            buy_index = buy_signals[i]
            if i < len(sell_signals):
                sell_index = sell_signals[i]
                money *= self.src[sell_index] / self.src[buy_index]
                profit.append(money - 100)
        self.profit  = pd.DataFrame(profit)
        self.rets = "Returns "+ self.tim +" = " + str(round(((money/100-1)*100),2)) + "%"
        plt.figure(figsize=(30,15))
        plt.plot(y2)
        plt.plot(self.src,color='black', label='close')
        for signal in buy_signals:
            plt.axvline(x=signal, color='green',linewidth=2)

        for signal in sell_signals:
            plt.axvline(x=signal, color='red',linewidth=2)
        plt.legend()
        plt.show()

st.title("Backtesting Nadaraya Regerssor")
ticks = pd.read_csv("100_tick.csv")

companies = ticks['Symbol'].values
selected_company = st.sidebar.selectbox("Select a company", companies)
tim = st.sidebar.text_input("time frame ex: 1y,2y,100d,10d..")
std = st.sidebar.text_input("Standard devation, like 5,6,7,8,9....")
backtest = Backtest(selected_company, tim, int(std))
st.pyplot(plt)
st.text(backtest.rets)
st.title("What is Nadaraya-Watson envelope?")
st.write("The Nadaraya-Watson envelope is a type of moving average calculated by taking a weighted average of data points over a period of time. The envelope is created by drawing two lines (one above and one below) parallel to the moving average at a user-defined percentage distance. Top and bottom lines form a “band” around the moving average that can be used to identify potential trend changes and generate trading signals. The Nadaraya-Watson cover is named after its creators, Vasily Nadaraya and his Geoffrey Watson.")
st.write(" [How we made this](https://share.streamlit.io/mesmith027/streamlit_webapps/main/MC_pi/streamlit_app.py)")