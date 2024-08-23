import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stock_prediction.data_handling import StockUtils

def fetch_indicators_for_symbols():
    symbols = ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'TSLA', 'GOOG', 'BRK.B', 'META', 'UNH', 'XOM', 'LLY', 'JPM', 'JNJ',  'PG', 'MA', 'AVGO', 'HD', 'CVX', 'MRK', 'COST', 'PEP', 'NFLX']
    technical_indicators = ["normalized_value", "2_reg", "3_reg", "5_reg", "10_reg", "20_reg", "50_reg", "adx", "ema", "sma", "rsi", "percent_b"]
    
    for symbol in symbols:
        # Instantiate the StockUtils class for the current symbol
        stock_utils = StockUtils(symbol=symbol)
        
        # Fetch the desired indicators
        df_stock_indicators = stock_utils.get_indicators(technical_indicators, min_max_order=10, outputsize=1200)
        
        print(f"Fetched indicators for {symbol}")

def main():
    fetch_indicators_for_symbols()

# This ensures that the main function runs when this script is executed directly
if __name__ == "__main__":
    main()
