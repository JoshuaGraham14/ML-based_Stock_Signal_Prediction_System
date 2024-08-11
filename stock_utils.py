import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from sklearn.linear_model import LinearRegression
from twelvedata import TDClient

class StockUtils:
    def __init__(self, symbol, interval="1day", outputsize=5000, config_path='config.json', json_dir='stock_data'):
        self.api_key = self.load_api_key(config_path)
        self.td_client = TDClient(apikey=self.api_key)
        self.json_dir = json_dir
        os.makedirs(self.json_dir, exist_ok=True)
        self.symbol = symbol
        self.interval = interval
        self.outputsize = outputsize
        self.df = None
        self.api_call_count = 0

        self.get_stock()

    def __str__(self):
        return f"StockUtils(symbol={self.symbol}, interval={self.interval}, outputsize={self.outputsize})"

    def load_api_key(self, config_path):
        """
        Load API key from a JSON configuration file.
        """
        with open(config_path) as config_file:
            config = json.load(config_file)
        return config['api_key']
    
    def track_api_call(self, fetch_data_string):
        """
        Increment the API call counter and print the fetch message.
        """
        self.api_call_count += 1
        print(f"Making API call to fetch {fetch_data_string}... (API call count: {self.api_call_count})")

    def get_stock(self):
        """
        Fetch stock data for a given symbol and save it to a JSON file.
        """
        today = datetime.now().strftime('%Y%m%d')
        json_filename = f'{self.symbol}_{today}_{self.interval}_{self.outputsize}.json'
        json_filepath = os.path.join(self.json_dir, json_filename)
        
        if os.path.exists(json_filepath):
            with open(json_filepath, 'r') as json_file:
                data = json.load(json_file)
            print(f"Loaded data from {json_filepath}")
        else:
            self.track_api_call(f"stock data for {self.symbol}")
            ts = self.td_client.time_series(symbol=self.symbol, interval=self.interval, outputsize=self.outputsize).as_json()
            data = self.transform_to_candle_list(ts)
            with open(json_filepath, 'w') as json_file:
                json.dump(data, json_file)
            print(f"Saved data to {json_filepath}")
        
        self.df = pd.DataFrame(data['candles'])

    def transform_to_candle_list(self, data):
        """
        Transform raw stock data into a candle list format.
        """
        candle_list = {"candles": []}
        for item in data:
            formatted_date = datetime.strptime(item['datetime'], "%Y-%m-%d").strftime("%d/%m/%Y")
            candle = {
                "date": formatted_date,
                "open": float(item["open"]),
                "high": float(item["high"]),
                "low": float(item["low"]),
                "close": float(item["close"]),
                "volume": int(item["volume"])
            }
            candle_list["candles"].append(candle)
        return candle_list

    def linear_regression(self, x, y):
        """
        Perform linear regression given x and y.
        """
        lr = LinearRegression()
        lr.fit(x, y)
        return lr.coef_[0][0]

    def n_day_regression(self, n):
        """
        n day regression for every data point.
        """
        var_name = f'{n}_reg'
        self.df[var_name] = np.nan

        for idx in range(n, len(self.df)):
            y = self.df['close'][idx - n: idx].to_numpy().reshape(-1, 1)
            x = np.arange(0, n).reshape(-1, 1)
            coef = self.linear_regression(x, y)
            self.df.loc[idx, var_name] = coef

    def normalized_values(self, high, low, close):
        """
        Normalize price between 0 and 1.
        """
        epsilon = 1e-10
        range_ = high - low
        close_range = close - low
        return close_range / (range_ + epsilon)

    def get_normalized(self):
        """
        Apply normalization to stock data.
        """
        self.df['normalized_value'] = self.df.apply(lambda x: self.normalized_values(x['high'], x['low'], x['close']), axis=1)

    def get_max_min(self, order=10):
        """
        Identify local minima and maxima in the stock data.
        """
        self.df['loc_min'] = self.df.iloc[argrelextrema(self.df['close'].values, np.less_equal, order=order)[0]]['close']
        self.df['loc_max'] = self.df.iloc[argrelextrema(self.df['close'].values, np.greater_equal, order=order)[0]]['close']
        
        self.idx_with_mins = np.where(self.df['loc_min'] > 0)[0]
        self.idx_with_maxs = np.where(self.df['loc_max'] > 0)[0]

    def get_adx(self):
        """
        Fetch ADX indicator data and add it to the DataFrame.
        """
        self.track_api_call(f"ADX indicator for {self.symbol}")
        adx = self.td_client.time_series(symbol=self.symbol, interval=self.interval, outputsize=self.outputsize).with_adx().as_pandas()
        self.df['adx'] = adx['adx'].values

    def get_ema(self, time_period=20):
        """
        Fetch EMA indicator data and add it to the DataFrame.
        """
        self.track_api_call(f"EMA indicator for {self.symbol} with time period {time_period}")
        ema = self.td_client.time_series(symbol=self.symbol, interval=self.interval, outputsize=self.outputsize).with_ema(time_period=time_period).as_pandas()
        self.df['ema'] = ema['ema'].values

    def get_percent_b(self):
        """
        Fetch Percent B indicator data and add it to the DataFrame.
        """
        self.track_api_call(f"Percent B indicator for {self.symbol}")
        percent_b = self.td_client.time_series(symbol=self.symbol, interval=self.interval, outputsize=self.outputsize).with_percent_b().as_pandas()
        self.df['percent_b'] = percent_b['percent_b'].values

    def get_rsi(self):
        """
        Fetch RSI indicator data and add it to the DataFrame.
        """
        self.track_api_call(f"RSI indicator for {self.symbol}")
        rsi = self.td_client.time_series(symbol=self.symbol, interval=self.interval, outputsize=self.outputsize).with_rsi().as_pandas()
        self.df['rsi'] = rsi['rsi'].values

    def get_sma(self, time_period=20):
        """
        Fetch SMA indicator data and add it to the DataFrame.
        """
        self.track_api_call(f"SMA indicator for {self.symbol} with time period {time_period}")
        sma = self.td_client.time_series(symbol=self.symbol, interval=self.interval, outputsize=self.outputsize).with_sma(time_period=time_period).as_pandas()
        self.df['sma'] = sma['sma'].values

    def get_indicators(self, technical_indicators):
        """
        Calculate specified indicators for the stock data.
        """
        self.get_max_min()

        if "normalized_value" in technical_indicators:
            self.get_normalized()
        
        regression_days = [int(ind.split('_')[0]) for ind in technical_indicators if ind.endswith('_reg')]
        if regression_days:
            for n in regression_days:
                self.n_day_regression(n)
        
        if "adx" in technical_indicators:
            self.get_adx()
        
        if "ema" in technical_indicators:
            self.get_ema()
        
        if "percent_b" in technical_indicators:
            self.get_percent_b()
        
        if "rsi" in technical_indicators:
            self.get_rsi()
        
        if "sma" in technical_indicators:
            self.get_sma()
        
        return self.df

    def plot_graph(self, show_mins=True, show_maxs=True):
        """
        Plot stock data with optional minima and maxima.
        """
        plt.figure(figsize=(10, 5))
        x = pd.to_datetime(self.df['date'], format='%d/%m/%Y')
        y = self.df["close"]
        plt.plot(x, y, ls='-', color="black", label="Daily")
        
        if show_mins:
            plt.plot(x, self.df['loc_min'], marker='o', color="green", linestyle='None', label="Local Minima")
        if show_maxs:
            plt.plot(x, self.df['loc_max'], marker='o', color="red", linestyle='None', label="Local Maxima")
        
        plt.xlabel("Date")
        plt.ylabel("Close (USD)")
        plt.legend()
        plt.title(f"{self.symbol}")
        plt.xticks(rotation=45)
        plt.show()