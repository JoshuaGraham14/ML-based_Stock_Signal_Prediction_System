import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

    def run(self, plot_graph=True):
        """
        Run the complete pipeline: train the model and test it with optional plotting.
        """
        model_trainer = ModelTrainer(
            self.training_symbols, 
            self.technical_indicators, 
            self.params)
        
        trained_model = model_trainer.train()

        model_predictor = ModelPredictor(
            trained_model,
            model_trainer.scaler,
            self.testing_symbols,
            self.technical_indicators,
            self.params
        )

        df_predictions = model_predictor.test(plot_graph=plot_graph)

        return df_predictions