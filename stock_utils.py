import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from sklearn.linear_model import LinearRegression
from twelvedata import TDClient

def load_api_key(config_path='config.json'):
    """
    Load API key from a JSON configuration file.
    """
    with open(config_path) as config_file:
        config = json.load(config_file)
    return config['api_key']

api_key = load_api_key()

td = TDClient(apikey=api_key)

JSON_DIR = 'stock_data'
os.makedirs(JSON_DIR, exist_ok=True)

def get_stock(symbol, interval="1day", outputsize=5000):
    """
    Fetch stock data for a given symbol and save it to a JSON file.
    """
    today = datetime.now().strftime('%Y%m%d')
    json_filename = f'{symbol}_{today}_{interval}_{outputsize}.json'
    json_filepath = os.path.join(JSON_DIR, json_filename)
    
    if os.path.exists(json_filepath):
        with open(json_filepath, 'r') as json_file:
            data = json.load(json_file)
        print(f"Loaded data from {json_filepath}")
    else:
        ts = td.time_series(symbol=symbol, interval=interval, outputsize=outputsize).as_json()
        data = transform_to_candle_list(ts)
        with open(json_filepath, 'w') as json_file:
            json.dump(data, json_file)
        print(f"Saved data to {json_filepath}")
    
    return pd.DataFrame(data['candles'])

def transform_to_candle_list(data):
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

def linear_regression(x, y):
    """
    Perform linear regression given x and y.
    """
    lr = LinearRegression()
    lr.fit(x, y)
    return lr.coef_[0][0]

def n_day_regression(n, df, idxs):
    """
    n day regression.
    """
    var_name = f'{n}_reg'
    df[var_name] = np.nan

    for idx in idxs:
        if idx > n:
            y = df['close'][idx - n: idx].to_numpy().reshape(-1, 1)
            x = np.arange(0, n).reshape(-1, 1)
            coef = linear_regression(x, y)
            df.loc[idx, var_name] = coef

    return df

def normalized_values(high, low, close):
    """
    Normalize price between 0 and 1.
    """
    epsilon = 1e-10
    range_ = high - low
    close_range = close - low
    return close_range / (range_ + epsilon)

def get_normalized(df):
    """
    Apply normalization to stock data.
    """
    df['normalized_value'] = df.apply(lambda x: normalized_values(x['high'], x['low'], x['close']), axis=1)
    return df

def get_max_min(df, order=10):
    """
    Identify local minima and maxima in the stock data.
    """
    df['loc_min'] = df.iloc[argrelextrema(df['close'].values, np.less_equal, order=order)[0]]['close']
    df['loc_max'] = df.iloc[argrelextrema(df['close'].values, np.greater_equal, order=order)[0]]['close']
    
    idx_with_mins = np.where(df['loc_min'] > 0)[0]
    idx_with_maxs = np.where(df['loc_max'] > 0)[0]
    
    return df, idx_with_mins, idx_with_maxs

def get_regressions(df, idx_with_mins, idx_with_maxs):
    """
    Calculate multiple n-day regressions for given indices.
    """
    for n in [2, 3, 5, 10, 20, 50]:
        df = n_day_regression(n, df, list(idx_with_mins) + list(idx_with_maxs))
    return df

def plot_graph(df, stock_symbol, show_mins=True, show_maxs=True):
    """
    Plot stock data with optional minima and maxima.
    """
    plt.figure(figsize=(10, 5))
    x = pd.to_datetime(df['date'], format='%d/%m/%Y')
    y = df["close"]
    plt.plot(x, y, ls='-', color="black", label="Daily")
    
    if show_mins:
        plt.plot(x, df['loc_min'], marker='o', color="green", linestyle='None', label="Local Minima")
    if show_maxs:
        plt.plot(x, df['loc_max'], marker='o', color="red", linestyle='None', label="Local Maxima")
    
    plt.xlabel("Date")
    plt.ylabel("Close (USD)")
    plt.legend()
    plt.title(f"{stock_symbol}")
    plt.xticks(rotation=45)
    plt.show()
