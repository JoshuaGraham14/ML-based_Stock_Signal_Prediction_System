import json
import os
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.linear_model import LinearRegression
from twelvedata import TDClient

class StockUtils:
    def __init__(self, symbol, interval="1day", outputsize=5000, config_path='config.json', json_dir='stock_data', min_max_order=10):
        self.api_key = self.load_api_key(config_path)
        self.td_client = TDClient(apikey=self.api_key)
        self.json_dir = json_dir
        os.makedirs(self.json_dir, exist_ok=True)
        self.symbol = symbol
        self.interval = interval
        self.outputsize = outputsize
        self.min_max_order = min_max_order
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

    def fetch_and_save_stock_data(self, json_filepath):
        """
        Helper function to fetch stock data and save the DataFrame with calculated indicators.
        """
        self.track_api_call(f"stock data for {self.symbol}")
        ts = self.td_client.time_series(symbol=self.symbol, interval=self.interval, outputsize=self.outputsize).as_json()
        data = self.transform_to_candle_list(ts)
        self.df = pd.DataFrame(data['candles'])

        # Convert 'date' to datetime if needed
        self.df['date'] = pd.to_datetime(self.df['date'], format='%d/%m/%Y')
        
        self.calculate_all_indicators()  # Calculate and store all indicators

        # Save the DataFrame to JSON
        self.df.to_json(json_filepath, orient='split')
        print(f"Saved DataFrame to {json_filepath}")

    def get_stock(self):
        """
        Fetch stock data for a given symbol and save the DataFrame with all calculated technical indicators.
        """
        today = datetime.now().strftime('%Y%m%d')
        json_filename = f'{self.symbol}_{today}_{self.interval}_{self.outputsize}.json'
        json_filepath = os.path.join(self.json_dir, json_filename)
        
        if os.path.exists(json_filepath):
            try:
                # Load the DataFrame from the JSON file
                self.df = pd.read_json(json_filepath, orient='split')

                # Ensure the 'date' column is in datetime format
                self.df['date'] = pd.to_datetime(self.df['date'], format='%d/%m/%Y')

                print(f"Loaded DataFrame from {json_filepath}")
            except ValueError as e:
                print(f"Error loading DataFrame: {e}. Re-fetching data and recalculating indicators.")
                self.fetch_and_save_stock_data(json_filepath)
        else:
            self.fetch_and_save_stock_data(json_filepath)

    def transform_to_candle_list(self, data):
        """
        Transform raw stock data into a candle list format.
        """
        candle_list = {"candles": []}
        for item in data:
            # Format the date to "05/08/2024" instead of "12\/08\/2024"
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

    def get_max_min(self):
        """
        Identify local minima and maxima in the stock data.
        """
        self.df['loc_min'] = self.df.iloc[argrelextrema(self.df['close'].values, np.less_equal, order=self.min_max_order)[0]]['close']
        self.df['loc_max'] = self.df.iloc[argrelextrema(self.df['close'].values, np.greater_equal, order=self.min_max_order)[0]]['close']
        
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

    def calculate_all_indicators(self):
        """
        Calculate all possible indicators and store them in the DataFrame.
        This method will be used to pre-calculate indicators when fetching stock data.
        """
        self.get_max_min()
        self.get_normalized()
        
        # Automatically detect all possible regression days from the existing indicator methods
        regression_days = self.extract_regression_days()
        for n in regression_days:
            self.n_day_regression(n)
        
        self.get_adx()
        self.get_ema()
        self.get_sma()
        # self.get_percent_b()
        # self.get_rsi()

        # Apply scaling to each feature individually
        # self.scale_features()

    def extract_regression_days(self):
        """
        Extract regression days from the current class methods.
        """
        regression_days = []
        for attr in dir(self):
            if attr.endswith('_reg') and attr.split('_')[0].isdigit():
                regression_days.append(int(attr.split('_')[0]))
        return sorted(set(regression_days))

    def get_indicators(self, technical_indicators):
        """
        Return specified indicators for the stock data.
        If the requested indicators are not calculated, they will be calculated on the fly.
        """
        # Identify missing indicators that are not in the current DataFrame
        missing_indicators = [ind for ind in technical_indicators if ind not in self.df.columns]
        
        # Calculate missing indicators if necessary
        if missing_indicators:
            if "normalized_value" in missing_indicators:
                self.get_normalized()
            regression_days = [int(ind.split('_')[0]) for ind in missing_indicators if ind.endswith('_reg')]
            if regression_days:
                for n in regression_days:
                    self.n_day_regression(n)
            if "adx" in missing_indicators:
                self.get_adx()
            if "ema" in missing_indicators:
                self.get_ema()
            if "percent_b" in missing_indicators:
                self.get_percent_b()
            if "rsi" in missing_indicators:
                self.get_rsi()
            if "sma" in missing_indicators:
                self.get_sma()
            
            # Save the updated DataFrame with the new indicators to JSON
            today = datetime.now().strftime('%Y%m%d')
            json_filename = f'{self.symbol}_{today}_{self.interval}_{self.outputsize}.json'
            json_filepath = os.path.join(self.json_dir, json_filename)
            self.df.to_json(json_filepath, orient='split')
            print(f"Updated DataFrame with new indicators saved to {json_filepath}")

        return self.df
    
    def scale_features(self):
        """
        Scale each feature individually using the most appropriate scaler.
        """
        scalers = {
            'normalized_value': MinMaxScaler(),
            'adx': MinMaxScaler(),
            'ema': StandardScaler(),
            'sma': StandardScaler()
        }

        # Apply RobustScaler to all regression features (e.g., '2_reg', '3_reg', etc.)
        regression_columns = [col for col in self.df.columns if col.endswith('_reg')]
        for col in regression_columns:
            scaler = RobustScaler()
            self.df[col] = scaler.fit_transform(self.df[[col]])

        # Apply scalers to specific features
        for feature, scaler in scalers.items():
            if feature in self.df.columns:
                self.df[feature] = scaler.fit_transform(self.df[[feature]])

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