import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stock_prediction.data_handling import StockUtils
from stock_prediction.models import StockPredictionModel

def main():
    training_symbols = ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'TSLA', 'GOOG', 'BRK.B', 'META', 'UNH', 'XOM', 'LLY', 'JPM', 'JNJ',  'PG', 'MA', 'AVGO', 'HD', 'CVX', 'MRK', 'COST', 'PEP', 'NFLX']
    testing_symbols = ['GOOG']
    technical_indicators = ["normalized_value", "2_reg", "3_reg", "5_reg", "10_reg", "20_reg", "50_reg", "adx", "ema", "sma", "rsi", "percent_b"]
    
    params = {
        'outputsize': 1000,
        'min_max_order': 5,
        'min_threshold': 0.0001,
        'max_threshold': 0.9999,
        'window_size': 2
    }

    stock = StockUtils('GOOG')
    stock.get_indicators(technical_indicators, outputsize=1000)
    stock.plot_graph()

    model = StockPredictionModel(training_symbols, testing_symbols, technical_indicators, params)
    model.run()

if __name__ == "__main__":
    main()