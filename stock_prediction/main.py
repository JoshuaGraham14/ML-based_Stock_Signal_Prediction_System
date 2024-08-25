import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Add the parent directory to sys.path

from stock_prediction.stock_predictor_pipeline import StockPredictorPipeline

def main():
    training_symbols = [
        'AAPL', 'MSFT', 'AMZN', 'NVDA', 'TSLA','BRK.B',
        'META', 'UNH', 'XOM', 'LLY', 'JPM', 'JNJ', 'PG', 'MA', 'AVGO', 'HD',
        'CVX', 'MRK', 'COST', 'PEP', 'NFLX'
    ]
    testing_symbols = ['GOOG']
    technical_indicators = [
        "normalized_value", "2_reg", "3_reg", "5_reg", "10_reg", "20_reg", 
        "50_reg", "adx", "ema", "sma", "rsi", "percent_b"
    ]

    params = {
        'outputsize': 1200,
        'min_max_order': 5,
        'min_threshold': 0.0001,
        'max_threshold': 0.9999,
        'window_size': 10
    }

    pipeline = StockPredictorPipeline(training_symbols, testing_symbols, technical_indicators, params)
    # pipeline.run()
    pipeline.run_backtest(initial_capital=10000, sell_perc=0.04, hold_till=5, stop_perc=0.0005)

if __name__ == "__main__":
    main()
