import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LinearRegression

from .api_handler import APIHandler

class StockUtils:
    # Define the list of technical indicators as a class property
    technical_indicators = ["normalized_value", "2_reg", "3_reg", "5_reg", "10_reg", "20_reg", "50_reg", "adx", "ema", "sma", "rsi", "percent_b"]

    def __init__(self, symbol, interval="1day", config_path='config.json', json_dir='data/stock_data'):
        """
        Initialize the StockUtils class with stock parameters and API handler.
        """
        self.api_key = self.load_api_key(config_path)
        self.api_handler = APIHandler(self.api_key)
        self.json_dir = json_dir
        os.makedirs(self.json_dir, exist_ok=True)
        self.symbol = symbol
        self.interval = interval
        self.outputsize = 5000
        self.df = None

        self.get_stock()

    def __str__(self):
        """
        Return a string representation of the StockUtils instance.
        """
        return f"StockUtils(symbol={self.symbol}, interval={self.interval}, outputsize={self.outputsize})"

    def load_api_key(self, config_path):
        """
        Load API key from a JSON configuration file.
        """
        with open(config_path) as config_file:
            config = json.load(config_file)
        return config['api_key']

    def fetch_and_save_stock_data(self, json_filepath):
        """
        Fetch stock data from API and save it to a JSON file.
        """
        def fetch_stock_data():
            return self.api_handler.td_client.time_series(
                symbol=self.symbol, 
                interval=self.interval, 
                outputsize=self.outputsize
            ).as_json()

        # Pass the symbol to the make_api_call method
        ts = self.api_handler.make_api_call(fetch_stock_data, symbol=self.symbol)
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
        Perform n-day linear regression for each data point.
        """
        # Reverse the DataFrame to get the data in chronological order
        df_reversed = self.df[::-1].reset_index(drop=True)
        
        var_name = f'{n}_reg'
        df_reversed[var_name] = np.nan

        for idx in range(n, len(df_reversed)):
            y = df_reversed['close'][idx - n: idx].to_numpy().reshape(-1, 1)
            x = np.arange(0, n).reshape(-1, 1)
            coef = self.linear_regression(x, y)
            df_reversed.loc[idx, var_name] = coef

        # Reverse the DataFrame back to the original order
        self.df[var_name] = df_reversed[var_name][::-1].reset_index(drop=True)

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

    def get_max_min(self, min_max_order):
        """
        Identify local minima and maxima in the stock data.
        """
        self.df['loc_min'] = self.df.iloc[argrelextrema(self.df['close'].values, np.less_equal, order=min_max_order)[0]]['close']
        self.df['loc_max'] = self.df.iloc[argrelextrema(self.df['close'].values, np.greater_equal, order=min_max_order)[0]]['close']
        
        self.idx_with_mins = np.where(self.df['loc_min'] > 0)[0]
        self.idx_with_maxs = np.where(self.df['loc_max'] > 0)[0]

    def get_adx(self):
        """
        Fetch and return ADX indicator data.
        """
        def fetch_adx():
            return self.api_handler.td_client.time_series(
                symbol=self.symbol, 
                interval=self.interval, 
                outputsize=self.outputsize
            ).with_adx().as_pandas()

        # Pass the symbol to the make_api_call method
        adx = self.api_handler.make_api_call(fetch_adx, symbol=self.symbol)
        self.df['adx'] = adx['adx'].values

    def get_ema(self, time_period=20):
        """
        Fetch and return EMA indicator data.
        """
        def fetch_ema():
            return self.api_handler.td_client.time_series(
                symbol=self.symbol, 
                interval=self.interval, 
                outputsize=self.outputsize
            ).with_ema(time_period=time_period).as_pandas()

        # Pass the symbol to the make_api_call method
        ema = self.api_handler.make_api_call(fetch_ema, symbol=self.symbol)
        self.df['ema'] = ema['ema'].values

    def get_percent_b(self):
        """
        Fetch and return Percent B indicator data.
        """
        def fetch_percent_b():
            return self.api_handler.td_client.time_series(
                symbol=self.symbol,
                interval=self.interval,
                outputsize=self.outputsize
            ).with_percent_b().as_pandas()

        # Pass the symbol to the make_api_call method
        percent_b = self.api_handler.make_api_call(fetch_percent_b, symbol=self.symbol)
        self.df['percent_b'] = percent_b['percent_b'].values

    def get_rsi(self):
        """
        Fetch and return RSI indicator data.
        """
        def fetch_rsi():
            return self.api_handler.td_client.time_series(
                symbol=self.symbol,
                interval=self.interval,
                outputsize=self.outputsize
            ).with_rsi().as_pandas()

        # Pass the symbol to the make_api_call method
        rsi = self.api_handler.make_api_call(fetch_rsi, symbol=self.symbol)
        self.df['rsi'] = rsi['rsi'].values

    def get_sma(self, time_period=20):
        """
        Fetch and return SMA indicator data.
        """
        def fetch_sma():
            return self.api_handler.td_client.time_series(
                symbol=self.symbol,
                interval=self.interval,
                outputsize=self.outputsize
            ).with_sma(time_period=time_period).as_pandas()

        # Pass the symbol to the make_api_call method
        sma = self.api_handler.make_api_call(fetch_sma, symbol=self.symbol)
        self.df['sma'] = sma['sma'].values

    def calculate_all_indicators(self):
        """
        Calculate all indicators listed in the class property and store them in the DataFrame.
        This method will be used to pre-calculate indicators when fetching stock data.
        """
        if "normalized_value" in self.technical_indicators:
            self.get_normalized()
        
        # Automatically detect and calculate all regression days based on the indicators list
        regression_days = [int(ind.split('_')[0]) for ind in self.technical_indicators if ind.endswith('_reg')]
        for n in regression_days:
            self.n_day_regression(n)
        
        if "adx" in self.technical_indicators:
            self.get_adx()
        if "ema" in self.technical_indicators:
            self.get_ema()
        if "sma" in self.technical_indicators:
            self.get_sma()
        if "percent_b" in self.technical_indicators:
            self.get_percent_b()
        if "rsi" in self.technical_indicators:
            self.get_rsi()

    def extract_regression_days(self):
        """
        Extract regression days from the current class methods.
        """
        regression_days = []
        for attr in dir(self):
            if attr.endswith('_reg') and attr.split('_')[0].isdigit():
                regression_days.append(int(attr.split('_')[0]))
        return sorted(set(regression_days))

    def get_indicators(self, technical_indicators, outputsize=5000, min_max_order=10, scale_features=False):
        """
        Return the DataFrame with the specified indicators, including "date", "open", "high", "low", "close", "volume", 
        "loc_min", and "loc_max". If any of the requested indicators are not valid, throw an error.
        This function also calculates local minima and maxima and scales the data if required.
        """
        # List of columns that should always be included
        required_columns = ["date", "open", "high", "low", "close", "volume", "loc_min", "loc_max"]
        
        # Validate requested indicators
        for indicator in technical_indicators:
            if indicator not in self.technical_indicators:
                raise ValueError(f"Invalid technical indicator requested: {indicator}")

        # Ensure all requested indicators are in the DataFrame
        missing_indicators = [ind for ind in technical_indicators if ind not in self.df.columns]
        if missing_indicators:
            raise ValueError(f"The following indicators are missing from the DataFrame: {', '.join(missing_indicators)}")

        # Call get_max_min to calculate local minima and maxima with the provided min_max_order
        self.get_max_min(min_max_order=min_max_order)

        # Optionally scale the features before returning the DataFrame
        if scale_features:
            self.scale_features()

        self.df = self.df.head(outputsize)

        # Return the DataFrame with the required columns and the requested indicators
        return self.df[required_columns + technical_indicators]
    
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
