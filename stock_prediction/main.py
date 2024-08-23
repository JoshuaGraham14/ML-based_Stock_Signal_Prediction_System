import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Add the parent directory to sys.path

from stock_prediction.data_handling import StockUtils
from stock_prediction.models import ModelTrainer
from stock_prediction.models import ModelPredictor

def run(training_symbols, testing_symbols, technical_indicators, params):
    model_trainer = ModelTrainer(training_symbols, technical_indicators, params)

    combined_df = model_trainer.gather_stock_data(training_symbols)
    training_df = model_trainer.prepare_training_data(combined_df)
    trained_model = model_trainer.train_model(training_df)

    model_predictor = ModelPredictor(
        trained_model,
        model_trainer.scaler, 
        testing_symbols,
        technical_indicators, 
        params
    )

    fig, axs = plt.subplots(len(testing_symbols), 1, figsize=(12, 6 * len(testing_symbols)))

    if len(testing_symbols) == 1:
        axs = [axs]  # Ensure axs is iterable when there's only one subplot

    for index, symbol in enumerate(testing_symbols):
        print(f"Processing stock: {symbol}")
        stock_utils_new = StockUtils(symbol=symbol)
        df_new_stock = stock_utils_new.get_indicators(technical_indicators, outputsize=params['outputsize'], min_max_order=params['min_max_order'])
        df_predictions = model_predictor.evaluate_model(df_new_stock)
        df_predictions = model_predictor.filter_predictions(df_predictions)
        model_predictor.plot_stock_predictions(axs, df_new_stock, df_predictions, symbol, index)

    plt.tight_layout()
    plt.show()

    return df_predictions

def main():
    training_symbols = ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'TSLA', 'GOOG', 'BRK.B', 'META', 'UNH', 'XOM', 'LLY', 'JPM', 'JNJ', 'PG', 'MA', 'AVGO', 'HD', 'CVX', 'MRK', 'COST', 'PEP', 'NFLX']
    testing_symbols = ['GOOG']
    technical_indicators = ["normalized_value", "2_reg", "3_reg", "5_reg", "10_reg", "20_reg", "50_reg", "adx", "ema", "sma", "rsi", "percent_b"]

    params = {
        'outputsize': 1200,
        'min_max_order': 5,
        'min_threshold': 0.0001,
        'max_threshold': 0.9999,
        'window_size': 2
    }

    run(training_symbols, testing_symbols, technical_indicators, params)

if __name__ == "__main__":
    main()
