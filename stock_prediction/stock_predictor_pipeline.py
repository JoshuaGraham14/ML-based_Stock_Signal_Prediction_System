import sys
import os
import matplotlib.pyplot as plt

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stock_prediction.data_handling import StockUtils
from stock_prediction.models import ModelTrainer, ModelPredictor

class StockPredictorPipeline:
    def __init__(self, training_symbols, testing_symbols, technical_indicators, params):
        """
        Initialize the StockPredictorPipeline class.

        Args:
        - training_symbols: List of stock symbols for training.
        - testing_symbols: List of stock symbols for testing.
        - technical_indicators: List of technical indicators to use.
        - params: Dictionary of parameters including 'outputsize', 'min_max_order', 'min_threshold',
                  'max_threshold', and 'window_size'.
        """
        self.training_symbols = training_symbols
        self.testing_symbols = testing_symbols
        self.technical_indicators = technical_indicators
        self.params = params

    def run(self):
        """
        Run the complete pipeline: gather data, train the model, and visualize predictions.
        """
        model_trainer = ModelTrainer(self.training_symbols, self.technical_indicators, self.params)

        combined_df = model_trainer.gather_stock_data(self.training_symbols)
        training_df = model_trainer.prepare_training_data(combined_df)
        trained_model = model_trainer.train_model(training_df)

        model_predictor = ModelPredictor(
            trained_model,
            model_trainer.scaler,
            self.testing_symbols,
            self.technical_indicators,
            self.params
        )

        fig, axs = plt.subplots(len(self.testing_symbols), 1, figsize=(12, 6 * len(self.testing_symbols)))

        if len(self.testing_symbols) == 1:
            axs = [axs]  # Ensure axs is iterable when there's only one subplot

        for index, symbol in enumerate(self.testing_symbols):
            print(f"Processing stock: {symbol}")
            stock_utils_new = StockUtils(symbol=symbol)
            df_new_stock = stock_utils_new.get_indicators(
                self.technical_indicators, 
                outputsize=self.params['outputsize'], 
                min_max_order=self.params['min_max_order']
            )
            df_predictions = model_predictor.evaluate_model(df_new_stock)
            df_predictions = model_predictor.filter_predictions(df_predictions)
            model_predictor.plot_stock_predictions(axs, df_new_stock, df_predictions, symbol, index)

        plt.tight_layout()
        plt.show()

        return df_predictions
    