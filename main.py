import time
from stock_utils import StockUtils
from StockPredictionModel import StockPredictionModel

# List of symbols and technical indicators
symbols = ['AAPL', 'WMT', 'MSFT', 'MA', 'AMZN', 'META', 'TSLA', 'GS', 'SPX', 'GOOG', 'NFLX']
technical_indicators = ["normalized_value", "2_reg", "3_reg", "5_reg", "10_reg", "20_reg", "50_reg", "adx", "ema", "sma"]

def fetch_indicators_for_symbols():
    for symbol in symbols:
        # Instantiate the StockUtils class for the current symbol
        stock_utils = StockUtils(symbol=symbol, outputsize=2000)
        
        # Fetch the desired indicators
        df_stock_indicators = stock_utils.get_indicators(technical_indicators)
        
        # Print or store the fetched indicators
        print(f"Fetched indicators for {symbol}")
        
        # Countdown before fetching the next symbol
        wait_time = 60  # Total wait time in seconds
        while wait_time > 0:
            time.sleep(10)
            wait_time -= 10
            print(f"Waiting... {wait_time} seconds left before fetching the next symbol.")
        print()  # Print a blank line for better readability

if __name__ == "__main__":
    # fetch_indicators_for_symbols()

    # Example usage
    training_symbols = ['AAPL', 'WMT', 'MSFT', 'MA', 'AMZN', 'META', 'TSLA', 'SPX', 'GOOG', 'NFLX']
    testing_symbols = ['NFLX', 'GOOG']
    technical_indicators = ["normalized_value", "2_reg", "3_reg", "5_reg", "10_reg", "20_reg", "50_reg", "adx", "ema", "sma"]

    model = StockPredictionModel(training_symbols, testing_symbols, technical_indicators, outputsize=2000)
    model.run()