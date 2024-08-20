from stock_utils import StockUtils
from StockPredictionModel import StockPredictionModel

# List of symbols and technical indicators
symbols = ['AAPL', 'WMT', 'MSFT', 'MA', 'AMZN', 'META', 'TSLA', 'GS', 'SPX', 'GOOG', 'NFLX']
# technical_indicators = ["normalized_value", "2_reg", "3_reg", "5_reg", "10_reg", "20_reg", "50_reg", "adx", "ema", "sma"]

def fetch_indicators_for_symbols():
    for symbol in symbols:
        # Instantiate the StockUtils class for the current symbol
        stock_utils = StockUtils(symbol=symbol)
        
        # # Fetch the desired indicators
        # df_stock_indicators = stock_utils.get_indicators(technical_indicators, min_max_order=10, outputsize=1200)

        # print(df_stock_indicators)

        # stock_utils.plot_graph()
        
        # Print or store the fetched indicators
        print(f"Fetched indicators for {symbol}")

if __name__ == "__main__":
    fetch_indicators_for_symbols()

    # Example usage
    training_symbols = ['AAPL', 'WMT', 'MSFT', 'MA', 'AMZN', 'META', 'TSLA', 'GS', 'SPX', 'GOOG', 'NFLX']
    testing_symbols = ['NFLX']
    technical_indicators = ["normalized_value", "2_reg", "3_reg", "5_reg", "10_reg", "20_reg", "50_reg", "adx", "ema", "sma", "rsi", "percent_b"]
    
    params = {
        'outputsize': 1200,
        'min_max_order': 10,
        'min_threshold': 0.0001,
        'max_threshold': 0.9999,
        'window_size': 2
    }

    model = StockPredictionModel(training_symbols, testing_symbols, technical_indicators, params)
    model.run()
